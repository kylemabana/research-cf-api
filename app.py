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
    'host': 'srv2051.hstgr.io', 
    'user': 'u311577524_admin', 
    'password': 'Ej@0MZ#*9', 
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
# CF PART (from recommend_cf.py) → /recommend endpoint
# ============================================================
def resolve_student_id(arg):
    """
    Resolves student ID from numeric student_id, student_number, or student name.
    
    If the input is not a number or student number format, it attempts to search 
    the student_information table by name.
    """
    raw = (arg or "").strip()
    if not raw: return None 

    conn = get_conn()
    try:
        # 1) Check if numeric -> Try as student_id
        if raw.isdigit():
            sid = int(raw)
            df = pd.read_sql(
                "SELECT student_id FROM student_information WHERE student_id = %s LIMIT 1",
                conn,
                params=[sid],
            )
            if not df.empty:
                # Safely get the ID
                return int(pd.to_numeric(df.iloc[0]["student_id"], errors="coerce"))


        # 2) Try as student_number (regardless of whether it's numeric or not, e.g., '22-00677')
        df = pd.read_sql(
            "SELECT student_id FROM student_information WHERE student_number = %s LIMIT 1",
            conn,
            params=[raw],
        )
        if not df.empty:
            # Safely get the ID
            return int(pd.to_numeric(df.iloc[0]["student_id"], errors="coerce"))

        # 3) Try as Student Name (for generic 'student' arg like 'John Doe')
        # We assume student_information has 'firstname' and 'lastname'
        if not raw.replace('-', '').isdigit(): 
            search_term = f"%{raw}%" # Use LIKE for partial matching
            
            # Search by name in separate query
            q_name = """
                SELECT student_id FROM student_information 
                WHERE CONCAT(firstname, ' ', lastname) LIKE %s OR firstname LIKE %s OR lastname LIKE %s 
                LIMIT 1
            """
            df_name = pd.read_sql(
                q_name,
                conn,
                params=[search_term, search_term, search_term],
            )

            if not df_name.empty:
                # Safely get the ID
                return int(pd.to_numeric(df_name.iloc[0]["student_id"], errors="coerce"))
            
        return None # No ID resolved
        
    except Exception as e:
        # Log the error but continue to return None
        print(f"Error resolving student ID for '{raw}': {e}")
        return None
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
        # CRUCIAL CHECK: Only attempt targeted fallback if program_id and college_id are valid
        if program_id and college_id and remaining > 0:
            # Dynamically build placeholders for exclusion list
            if exclude_ids:
                exclude_placeholders = ",".join(["%s"] * len(exclude_ids))
                exclude_clause = f"AND tc.tc_id NOT IN ({exclude_placeholders})"
                params1_ex = exclude_ids
            else:
                exclude_clause = ""
                params1_ex = []

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
            params = [program_id, college_id] + params1_ex + [remaining]
            df1 = pd.read_sql(q1, conn, params=params)
            frames.append(df1)
            remaining -= len(df1)

        # Fallback 2: most read overall (Approved only)
        if remaining > 0:
            # Collect all IDs already recommended (CF + Fallback 1)
            ex_ids = exclude_ids[:]
            for f in frames:
                if not f.empty:
                    ex_ids += f["tc_id"].tolist()
            ex_ids = list(set(ex_ids))

            if ex_ids:
                exclude_placeholders = ",".join(["%s"] * len(ex_ids))
                where_clause = f"WHERE tc.tc_id NOT IN ({exclude_placeholders})"
                params2_ex = ex_ids
            else:
                where_clause = ""
                params2_ex = []

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
            params2 = params2_ex + [remaining]
            df2 = pd.read_sql(q2, conn, params=params2)
            frames.append(df2)

        if frames:
            # Use .dropna() to remove any rows where tc_id might have failed conversion during DB read
            return pd.concat(frames, ignore_index=True).drop_duplicates(subset=['tc_id']).dropna(subset=['tc_id'])
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

        def safe_to_int(df, col):
            if df.empty or col not in df.columns: return None
            val = df.iloc[0].get(col)
            if pd.isna(val) or val is None: return None
            try: return int(val)
            except ValueError: return None
        
        program_id = safe_to_int(prog_col_df, "program_id")
        college_id = safe_to_int(prog_col_df, "colleges_id")
        
        # Identify items current user has already read (for exclusion later)
        current_items = set(reads_df[reads_df["student_id"] == student_id]["tc_id"])

        # --- CF Calculation Attempt ---
        recommend_df = pd.DataFrame()
        
        # If no reads at all or current student is not in the reads_df, skip CF
        if reads_df.empty or student_id not in reads_df['student_id'].unique():
            # Go straight to targeted fallback
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
    
    Usage: /recommend?student_id=3 (Preferred)
           /recommend?student_number=22-00677 
           /recommend?student=John Doe (Name Lookup)
    """
    # FIX: Explicitly setting the priority as requested: student_id > student_number > generic student
    student_arg = (
        request.args.get("student_id", "").strip() # 1. Highest Priority
        or request.args.get("student_number", "").strip() # 2. Secondary Priority
        or request.args.get("student", "").strip() # 3. Lowest Priority (Name/Generic Lookup)
    )

    if not student_arg:
        # Global fallback if no argument is provided
        fb = fallback_recos(limit=4)
        return jsonify(fb.to_dict(orient="records"))

    recs = compute_recommendations(student_arg)
    return jsonify(recs)


# ============================================================
# SEARCH PART → /search endpoint
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
# ENTRYPOINT
# ============================================================
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
