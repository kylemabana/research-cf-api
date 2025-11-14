import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify
import pymysql
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

app = Flask(__name__)

# ----------------------------
# MySQL connection config (EDIT THIS)
# ----------------------------
db_config = {
    'host': 'srv2051.hstgr.io',            # change to your DB host if needed
    'user': 'u311577524_admin',            # change to your DB user
    'password': 'Ej@0MZ#*9',               # change to your DB password
    'database': 'u311577524_research_db',
    'cursorclass': pymysql.cursors.DictCursor
}

def get_conn():
    """Establishes a MySQL connection using pymysql."""
    return pymysql.connect(**db_config)


def drop_header_like_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows na mukhang header (e.g. title='title', authorone='authorone', etc.).
    """
    if df is None or df.empty:
        return df

    cols = list(df.columns)

    def is_header_row(row):
        matches = 0
        for col in cols:
            val = row.get(col)
            if isinstance(val, str) and val.strip().lower() == col.strip().lower():
                matches += 1
        # kung maraming columns ang eksaktong kapangalan ng column, mukhang header
        return matches >= 2

    mask = df.apply(is_header_row, axis=1)
    # baliktarin: keep rows na HINDI header
    cleaned = df[~mask].copy()
    return cleaned


# ----------------------------
# Load vectorizer / model for /search
# ----------------------------
try:
    vectorizer = joblib.load('vectorizer.pkl')
    # If you don't actually use model, you can delete this line
    model = joblib.load('model.pkl')
except FileNotFoundError:
    # Handle case where pkl files are missing (e.g., development environment)
    print("Warning: vectorizer.pkl or model.pkl not found. Search functionality will fail.")
    vectorizer = None
    model = None


# ============================================================
#  CF PART (from recommend_cf.py) → /recommend endpoint
# ============================================================
def resolve_student_id(arg):
    """
    Resolves student ID from either numeric student_id or student_number.
    """
    raw = (arg or "").strip()

    conn = get_conn()
    try:
        # 1) Kung mukhang numeric, subukan muna as student_id mismo
        if raw.isdigit():
            sid = int(raw)
            df = pd.read_sql(
                "SELECT student_id FROM student_information WHERE student_id = %s LIMIT 1",
                conn,
                params=[sid],
            )
            if not df.empty:
                return sid  # valid student_id na

        # 2) Kung hindi nag-match as student_id, hanapin as student_number
        df = pd.read_sql(
            "SELECT student_id FROM student_information WHERE student_number = %s",
            conn,
            params=[raw],
        )

        if df.empty:
            return None

        # 3) Linisin kung may header / maling data (e.g. 'student_id')
        df["student_id"] = pd.to_numeric(df["student_id"], errors="coerce")
        df = df.dropna(subset=["student_id"])

        if df.empty:
            # lahat ng nakuha ay hindi valid na number (hal. 'student_id')
            return None

        # Take the first resolved ID
        return int(df.iloc[0]["student_id"])
    finally:
        conn.close()


def fallback_recos(program_id=None, college_id=None, exclude_ids=None, limit=4):
    """
    Generates fallback recommendations based on popularity (most read) among approved theses.
    Prioritizes same college/program, then global popularity.
    """
    exclude_ids = exclude_ids or []
    conn = get_conn()
    try:
        frames = []
        remaining = limit

        # Fallback 1: most read with same program & college (Approved only)
        if program_id and college_id and remaining > 0:
            # Dynamically build placeholders for exclusion list
            exclude_placeholders = ",".join(["%s"] * len(exclude_ids))
            exclude_clause = f"AND tc.tc_id NOT IN ({exclude_placeholders})" if exclude_ids else ""

            q1 = f"""
                SELECT tc.tc_id, tc.title, tc.authorone, tc.authortwo, tc.colleges_id, tc.program_id,
                       tc.academic_year, tc.project_type, COUNT(sr.read_id) AS read_count
                FROM thesis_capstone tc
                INNER JOIN thesis_submission ts ON ts.tc_id = tc.tc_id AND ts.status = 'Approved'
                LEFT JOIN student_reads sr ON sr.tc_id = tc.tc_id
                WHERE tc.program_id = %s AND tc.colleges_id = %s
                {exclude_clause}
                GROUP BY tc.tc_id
                ORDER BY read_count DESC, tc.tc_id DESC
                LIMIT %s
            """
            params = [program_id, college_id] + (exclude_ids if exclude_ids else []) + [remaining]
            df1 = pd.read_sql(q1, conn, params=params)
            frames.append(df1)
            remaining -= len(df1)

        # Fallback 2: most read overall (Approved only)
        if remaining > 0:
            # Collect all IDs already recommended (CF + Fallback 1)
            ex_ids = exclude_ids[:]
            for f in frames:
                if not f.empty:
                    # Use unique list in case of overlap
                    ex_ids += f["tc_id"].tolist()
            ex_ids = list(set(ex_ids)) 

            exclude_placeholders = ",".join(["%s"] * len(ex_ids))
            where_clause = f"WHERE tc.tc_id NOT IN ({exclude_placeholders})" if ex_ids else ""

            q2 = f"""
                SELECT tc.tc_id, tc.title, tc.authorone, tc.authortwo, tc.colleges_id, tc.program_id,
                       tc.academic_year, tc.project_type, COUNT(sr.read_id) AS read_count
                FROM thesis_capstone tc
                INNER JOIN thesis_submission ts ON ts.tc_id = tc.tc_id AND ts.status = 'Approved'
                LEFT JOIN student_reads sr ON sr.tc_id = tc.tc_id
                {where_clause}
                GROUP BY tc.tc_id
                ORDER BY read_count DESC, tc.tc_id DESC
                LIMIT %s
            """
            params2 = (ex_ids if ex_ids else []) + [remaining]
            df2 = pd.read_sql(q2, conn, params=params2)
            frames.append(df2)

        if frames:
            return pd.concat(frames, ignore_index=True).drop_duplicates(subset=['tc_id'])
        return pd.DataFrame()
    finally:
        conn.close()


def compute_recommendations(student_arg):
    # 1) Resolve to internal student_id 
    student_id = resolve_student_id(student_arg)

    # Kung hindi ma-resolve → global popular lang (Approved)
    if student_id is None:
        fb = fallback_recos(limit=4)
        return fb.to_dict(orient="records")

    conn = get_conn()
    try:
        # Get all read history for CF matrix creation
        reads_df = pd.read_sql("SELECT student_id, tc_id FROM student_reads", conn)
        
        # --- FIX: Clean and ensure numeric columns before building the CF matrix ---
        reads_df = drop_header_like_rows(reads_df)
        
        # Convert columns to numeric, coercing errors to NaN, then drop NaNs
        reads_df["student_id"] = pd.to_numeric(reads_df["student_id"], errors="coerce")
        reads_df["tc_id"] = pd.to_numeric(reads_df["tc_id"], errors="coerce")
        reads_df = reads_df.dropna(subset=["student_id", "tc_id"]).astype({'student_id': int, 'tc_id': int})

        # Get program/college for targeted fallback
        prog_col_df = pd.read_sql(
            "SELECT program_id, colleges_id FROM student_information WHERE student_id = %s LIMIT 1",
            conn,
            params=[student_id]
        )

        # Helper to safely convert to integer
        def safe_to_int(v):
            try:
                if pd.isna(v): return None
            except Exception: pass
            try: return int(v)
            except Exception:
                try: return int(float(str(v)))
                except Exception: return None

        program_id = safe_to_int(prog_col_df.iloc[0].get("program_id")) if not prog_col_df.empty else None
        college_id = safe_to_int(prog_col_df.iloc[0].get("colleges_id")) if not prog_col_df.empty else None
        
        # Identify items current user has already read (for exclusion later)
        # Using the cleaned reads_df now
        current_items = set(reads_df[reads_df["student_id"] == student_id]["tc_id"])

        # --- CF Calculation Attempt ---
        recommend_df = pd.DataFrame()
        
        # If no reads at all or current student is not in the reads_df, skip CF
        if reads_df.empty or student_id not in reads_df['student_id'].unique():
            # Go straight to fallback based on student's program/college
            rec_df = fallback_recos(program_id=program_id, college_id=college_id, limit=4)
            rec_df = drop_header_like_rows(rec_df)
            return rec_df.to_dict(orient="records")

        # Build user-item matrix
        user_item = pd.crosstab(reads_df["student_id"], reads_df["tc_id"])
        user_item.index = user_item.index.astype(int)
        user_item.columns = user_item.columns.astype(int)

        # Recalculate if user is still missing after crosstab (edge case handling)
        if int(student_id) not in user_item.index:
            rec_df = fallback_recos(program_id=program_id, college_id=college_id, limit=4)
            rec_df = drop_header_like_rows(rec_df)
            return rec_df.to_dict(orient="records")

        # Calculate user-user Cosine similarity
        sim = cosine_similarity(user_item.values)
        sim_df = pd.DataFrame(sim, index=user_item.index, columns=user_item.index)

        # Find top 5 similar students (excluding self)
        similar_students = sim_df[student_id].sort_values(ascending=False).iloc[1:6].index.tolist()
        candidate_ids = []

        if similar_students:
            similar_reads = reads_df[reads_df["student_id"].isin(similar_students)]
            
            # Count how many similar students read each thesis (for ranking)
            candidate_counts = (
                similar_reads.groupby("tc_id")["student_id"]
                .nunique() 
                .sort_values(ascending=False)
            )
            
            # Select candidates that the current user hasn't read
            candidate_ids = [
                int(tc) for tc in candidate_counts.index
                if tc not in current_items
            ]

        if candidate_ids:
            # Fetch details only for APPROVED theses from the candidates
            placeholders = ",".join(["%s"] * len(candidate_ids))
            recommend_query = f"""
                SELECT tc.tc_id, tc.title, tc.authorone, tc.authortwo,
                       tc.colleges_id, tc.program_id, tc.academic_year, tc.project_type
                FROM thesis_capstone tc
                INNER JOIN thesis_submission ts ON ts.tc_id = tc.tc_id AND ts.status = 'Approved'
                WHERE tc.tc_id IN ({placeholders})
            """
            recommend_df = pd.read_sql(recommend_query, conn, params=candidate_ids)

            if not recommend_df.empty:
                # Map the CF ranking back to the DataFrame and sort
                recommend_df["tc_id"] = recommend_df["tc_id"].astype(int)
                rank_map = {tc_id: rank for rank, tc_id in enumerate(candidate_ids)}
                recommend_df["cf_rank"] = recommend_df["tc_id"].map(rank_map).fillna(10_000)
                recommend_df = recommend_df.sort_values(["cf_rank", "tc_id"]).drop(columns=["cf_rank"])
                recommend_df = recommend_df.head(4)

        # --- Fallback to fill up to 4 recommendations ---
        if len(recommend_df) < 4:
            remaining = 4 - len(recommend_df)
            
            # Exclude current CF results AND items user has already read
            exclude_ids = recommend_df["tc_id"].tolist() if not recommend_df.empty else []
            exclude_ids = list(set(exclude_ids + list(current_items))) # Combine CF results and user history

            # Call fallback_recos (which internally does targeted and global popular)
            fb = fallback_recos(
                program_id=program_id, # Attempt targeted fallback first
                college_id=college_id,
                exclude_ids=exclude_ids,
                limit=remaining
            )
            if not fb.empty:
                recommend_df = pd.concat([recommend_df, fb], ignore_index=True)
                # Ensure we only return max of 4 total recommendations
                recommend_df = recommend_df.head(4) 


        # FINAL SAFETY: alisin lahat ng mukhang header rows
        recommend_df = drop_header_like_rows(recommend_df)

        # Final output format
        return recommend_df.to_dict(orient="records")
    finally:
        conn.close()


@app.route("/recommend")
def recommend():
    """
    Endpoint to retrieve thesis recommendations based on Collaborative Filtering (CF)
    or popular fallbacks for a given student.
    Usage: /recommend?student_id=3 or /recommend?student_number=22-00677
    """
    # primary: student_number, fallback: student_id or generic 'student'
    student_arg = (
        request.args.get("student_number", "").strip()
        or request.args.get("student", "").strip()
        or request.args.get("student_id", "").strip()
    )

    if not student_arg:
        # Global fallback if no argument is provided
        fb = fallback_recos(limit=4)
        return jsonify(fb.to_dict(orient="records"))

    recs = compute_recommendations(student_arg)
    return jsonify(recs)


# ============================================================
#  SEARCH PART → /search endpoint
# ============================================================
@app.route('/search', methods=['POST'])
def search():
    data = request.get_json(silent=True) or {}
    query = data.get('query', '').strip()
    if not query:
        return jsonify([])

    if vectorizer is None:
        # Error handling if model wasn't loaded
        return jsonify({"error": "Search model not loaded on the server."}), 500

    conn = get_conn()
    try:
        with conn.cursor() as cursor:
            # Query all approved theses with authors and college/program details
            cursor.execute("""
                SELECT tc.tc_id, tc.title, tc.authorone, tc.authortwo, tc.authorthree,
                       tc.colleges_id, p.program, c.colleges AS college
                FROM thesis_capstone tc
                INNER JOIN thesis_submission ts ON ts.tc_id = tc.tc_id AND ts.status = 'Approved'
                LEFT JOIN student_information si ON tc.student_id = si.student_id 
                LEFT JOIN program p ON tc.program_id = p.program_id
                LEFT JOIN colleges c ON tc.colleges_id = c.colleges_id
            """)
            rows = cursor.fetchall()
    finally:
        conn.close()

    df = pd.DataFrame(rows)
    if df.empty:
        return jsonify([])

    # Combine relevant columns for vectorization
    df['text'] = df[['title', 'authorone', 'authortwo', 'authorthree']].fillna('').agg(' '.join, axis=1)

    # Calculate similarity
    corpus_vec = vectorizer.transform(df['text'])
    q_vec = vectorizer.transform([query])
    scores = (corpus_vec @ q_vec.T).toarray().flatten()

    df['score'] = scores
    # Take the top 5 results based on similarity score
    top = df.nlargest(5, 'score')
    
    # Format results
    results = top[['tc_id', 'title', 'college', 'program', 'score']].to_dict(orient='records')
    return jsonify(results)

# ============================================================
#  ENTRYPOINT
# ============================================================
if __name__ == '__main__':
    # When deploying on a platform, you might not want debug=True

    app.run(host="0.0.0.0", port=8000, debug=True)