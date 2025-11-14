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
    'host': 'srv2051.hstgr.io',          # change to your DB host if needed
    'user': 'u311577524_admin',               # change to your DB user
    'password': 'Ej@0MZ#*9',               # change to your DB password
    'database': 'u311577524_research_db',
    'cursorclass': pymysql.cursors.DictCursor
}

def get_conn():
    return pymysql.connect(**db_config)

# ----------------------------
# Load vectorizer / model for /search
# ----------------------------
vectorizer = joblib.load('vectorizer.pkl')
# If you don't actually use model, you can delete this line
model = joblib.load('model.pkl')

# ============================================================
#  CF PART (from recommend_cf.py) → /recommend endpoint
# ============================================================

def resolve_student_id(arg):
    raw = (arg or "").strip()
    if raw.isdigit():
        return int(raw)

    conn = get_conn()
    try:
        df = pd.read_sql(
            "SELECT student_id FROM student_information WHERE student_number = %s LIMIT 1",
            conn,
            params=[raw]
        )
        if not df.empty:
            return int(df.iloc[0]["student_id"])
        return None
    finally:
        conn.close()

def fallback_recos(program_id=None, college_id=None, exclude_ids=None, limit=4):
    exclude_ids = exclude_ids or []
    conn = get_conn()
    try:
        frames = []
        remaining = limit

        # Fallback 1: most read with same program & college (Approved only)
        if program_id and college_id and remaining > 0:
            q1 = f"""
                SELECT tc.tc_id, tc.title, tc.authorone, tc.authortwo, tc.colleges_id, tc.program_id,
                       tc.academic_year, tc.project_type, COUNT(sr.read_id) AS read_count
                FROM thesis_capstone tc
                INNER JOIN thesis_submission ts ON ts.tc_id = tc.tc_id AND ts.status = 'Approved'
                LEFT JOIN student_reads sr ON sr.tc_id = tc.tc_id
                WHERE tc.program_id = %s AND tc.colleges_id = %s
                {("AND tc.tc_id NOT IN (" + ",".join(["%s"]*len(exclude_ids)) + ")") if exclude_ids else ""}
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
            ex_ids = exclude_ids[:]
            for f in frames:
                if not f.empty:
                    ex_ids += f["tc_id"].tolist()

            where_clause = ""
            if ex_ids:
                where_clause = "WHERE tc.tc_id NOT IN (" + ",".join(["%s"] * len(ex_ids)) + ")"

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
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame()
    finally:
        conn.close()

def compute_recommendations(student_arg):
    student_id = resolve_student_id(student_arg)

    # If cannot resolve → global popular/approved
    if not student_id:
        fb = fallback_recos(limit=4)
        return fb.to_dict(orient="records")

    conn = get_conn()
    try:
        reads_df = pd.read_sql("SELECT student_id, tc_id FROM student_reads", conn)
        prog_col_df = pd.read_sql(
            "SELECT program_id, colleges_id FROM student_information WHERE student_id = %s LIMIT 1",
            conn,
            params=[student_id]
        )

        def safe_to_int(v):
            try:
                if pd.isna(v):
                    return None
            except Exception:
                pass
            try:
                # handle numeric strings, floats stored as strings, etc.
                return int(v)
            except Exception:
                try:
                    return int(float(str(v)))
                except Exception:
                    return None

        if not prog_col_df.empty:
            program_id = safe_to_int(prog_col_df.iloc[0].get("program_id"))
            college_id = safe_to_int(prog_col_df.iloc[0].get("colleges_id"))
        else:
            program_id = None
            college_id = None

        if reads_df.empty:
            fb = fallback_recos(program_id=program_id, college_id=college_id, limit=4)
            return fb.to_dict(orient="records")

        # build binary user-item matrix
        user_item = pd.crosstab(reads_df["student_id"].astype(int), reads_df["tc_id"].astype(int))
        user_item.index = user_item.index.astype(int)
        user_item.columns = user_item.columns.astype(int)

        if int(student_id) not in user_item.index:
            fb = fallback_recos(program_id=program_id, college_id=college_id, limit=4)
            return fb.to_dict(orient="records")

        sim = cosine_similarity(user_item.values)
        sim_df = pd.DataFrame(sim, index=user_item.index, columns=user_item.index)

        similar_students = sim_df[student_id].sort_values(ascending=False).iloc[1:6].index.tolist()

        current_items = set(reads_df[reads_df["student_id"] == student_id]["tc_id"])
        recommend_df = pd.DataFrame()
        candidate_ids = []

        if similar_students:
            similar_reads = reads_df[reads_df["student_id"].isin(similar_students)]
            candidate_counts = (
                similar_reads.groupby("tc_id")["student_id"]
                .nunique()
                .sort_values(ascending=False)
            )
            candidate_ids = [
                int(tc) for tc in candidate_counts.index
                if tc not in current_items
            ]

        if candidate_ids:
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
                recommend_df["tc_id"] = recommend_df["tc_id"].astype(int)
                rank_map = {tc_id: rank for rank, tc_id in enumerate(candidate_ids)}
                recommend_df["cf_rank"] = recommend_df["tc_id"].map(rank_map).fillna(10_000)
                recommend_df = recommend_df.sort_values(["cf_rank", "tc_id"]).drop(columns=["cf_rank"])
                recommend_df = recommend_df.head(4)

        if len(recommend_df) < 4:
            remaining = 4 - len(recommend_df)
            exclude_ids = recommend_df["tc_id"].tolist() if not recommend_df.empty else []
            fb = fallback_recos(
                program_id=program_id,
                college_id=college_id,
                exclude_ids=exclude_ids,
                limit=remaining
            )
            if not fb.empty:
                recommend_df = pd.concat([recommend_df, fb], ignore_index=True)

        return recommend_df.to_dict(orient="records")
    finally:
        conn.close()

@app.route("/recommend")
def recommend():
    student_arg = request.args.get("student_id", "").strip()
    if not student_arg:
        return jsonify([])
    recs = compute_recommendations(student_arg)
    return jsonify(recs)

# ============================================================
#  SEARCH PART → /search endpoint
# ============================================================
@app.route('/search', methods=['POST'])
def search():
    data = request.get_json(silent=True) or {}
    query = data.get('query', '').strip()
    if not query:
        return jsonify([])

    conn = get_conn()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT tc.title, tc.authorone, tc.authortwo, tc.authorthree,
                       tc.colleges_id, p.program, c.colleges AS college
                FROM thesis_capstone tc
                JOIN thesis_submission ts USING(tc_id)
                JOIN student_information si USING(student_id)
                JOIN program p USING(program_id)
                JOIN colleges c ON si.colleges_id = c.colleges_id
                WHERE ts.status = 'Approved'
            """)
            rows = cursor.fetchall()
    finally:
        conn.close()

    df = pd.DataFrame(rows)
    if df.empty:
        return jsonify([])

    df['text'] = df[['title', 'authorone', 'authortwo', 'authorthree']].fillna('').agg(' '.join, axis=1)

    corpus_vec = vectorizer.transform(df['text'])
    q_vec = vectorizer.transform([query])
    scores = (corpus_vec @ q_vec.T).toarray().flatten()

    df['score'] = scores
    top = df.nlargest(5, 'score')
    results = top[['title', 'college', 'program']].to_dict(orient='records')
    return jsonify(results)

# ============================================================
#  ENTRYPOINT
# ============================================================
if __name__ == '__main__':
    # When deploying on a platform, you might not want debug=True
    app.run(host="0.0.0.0", port=8000, debug=True)
