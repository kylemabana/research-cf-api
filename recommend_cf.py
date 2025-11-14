import sys
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import mysql.connector
import os
import json

# ----------------------------
# DB connection helper
# ----------------------------
def get_conn():
    # Read DB config from .dbconfig.json if present, else from environment, else fallbacks
    cfg = {}
    cfg_path = os.path.join(os.path.dirname(__file__), '.dbconfig.json')
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}

    host = cfg.get('host') or os.environ.get('DB_HOST') or 'srv2051.hstgr.io'
    user = cfg.get('user') or os.environ.get('DB_USER') or 'u311577524_admin'
    password = cfg.get('password') or os.environ.get('DB_PASSWORD') or 'Ej@0MZ#*9'
    database = cfg.get('database') or os.environ.get('DB_NAME') or 'u311577524_research_db'
    port = int(cfg.get('port') or os.environ.get('DB_PORT') or 3306)

    return mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        port=port
    )

# ----------------------------
# Resolve incoming arg to numeric student_id
# Accepts either numeric student_id or student_number like '22-00677'
# ----------------------------
def resolve_student_id(arg):
    raw = (arg or "").strip()
    if raw.isdigit():
        return int(raw)

    # Try to resolve as student_number from student_information
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

# ----------------------------
# Popular/Approved fallbacks
# ----------------------------
def fallback_recos(program_id=None, college_id=None, exclude_ids=None, limit=4):
    exclude_ids = exclude_ids or []
    conn = get_conn()
    try:
        # Fallback 1: most read with same program & college (Approved only)
        frames = []
        remaining = limit

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

            q2 = f"""
                SELECT tc.tc_id, tc.title, tc.authorone, tc.authortwo, tc.colleges_id, tc.program_id,
                       tc.academic_year, tc.project_type, COUNT(sr.read_id) AS read_count
                FROM thesis_capstone tc
                INNER JOIN thesis_submission ts ON ts.tc_id = tc.tc_id AND ts.status = 'Approved'
                LEFT JOIN student_reads sr ON sr.tc_id = tc.tc_id
                {("WHERE tc.tc_id NOT IN (" + ",".join(["%s"]*len(ex_ids)) + ")") if ex_ids else ""}
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

def main():
    # No arg? Return empty (or you could return global fallback)
    if len(sys.argv) < 2:
        print(json.dumps([]))
        return

    # Resolve student id
    arg = sys.argv[1]
    student_id = resolve_student_id(arg)

    # If we still can't resolve, just return global popular approved
    if not student_id:
        fb = fallback_recos(limit=4)
        print(fb.to_json(orient="records", force_ascii=False))
        return

    conn = get_conn()
    try:
        # Read history
        reads_df = pd.read_sql("SELECT student_id, tc_id FROM student_reads", conn)
        # program/college for targeted fallback
        prog_col_df = pd.read_sql(
            "SELECT program_id, colleges_id FROM student_information WHERE student_id = %s LIMIT 1",
            conn,
            params=[student_id]
        )
        program_id = int(prog_col_df.iloc[0]["program_id"]) if not prog_col_df.empty and pd.notna(prog_col_df.iloc[0]["program_id"]) else None
        college_id = int(prog_col_df.iloc[0]["colleges_id"]) if not prog_col_df.empty and pd.notna(prog_col_df.iloc[0]["colleges_id"]) else None

        # If no reads at all → fallback
        if reads_df.empty:
            fb = fallback_recos(program_id=program_id, college_id=college_id, limit=4)
            print(fb.to_json(orient="records", force_ascii=False))
            return

        # Build user-item matrix (rows=student_id, cols=tc_id) as binary counts
        user_item = pd.crosstab(reads_df["student_id"].astype(int), reads_df["tc_id"].astype(int))
        # Ensure indices/columns are integer types for reliable lookups later
        user_item.index = user_item.index.astype(int)
        user_item.columns = user_item.columns.astype(int)

        # If current user missing in matrix → fallback
        if int(student_id) not in user_item.index:
            fb = fallback_recos(program_id=program_id, college_id=college_id, limit=4)
            print(fb.to_json(orient="records", force_ascii=False))
            return

        # Cosine similarity
        sim = cosine_similarity(user_item.values)
        sim_df = pd.DataFrame(sim, index=user_item.index, columns=user_item.index)

        # Similar students (top 5 excluding self)
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

        # Fill to 4 with fallbacks (Approved only)
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

        # Final JSON
        print(recommend_df.to_json(orient="records", force_ascii=False))
    finally:
        conn.close()

if __name__ == "__main__":
    main()
