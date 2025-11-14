from fastapi import FastAPI
from typing import List, Dict, Any
import pymysql

app = FastAPI()


def get_db():
    # 🔴 EDIT THESE CREDENTIALS TO MATCH YOUR DATABASE
    return pymysql.connect(
        host="srv2051.hstgr.io",
        user="u311577524_admin",
        password="Ej@0MZ#*9",
        database="u311577524_research_db",
        cursorclass=pymysql.cursors.DictCursor
    )


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/recommend")
def recommend(student_id: int) -> List[Dict[str, Any]]:
    # ✅ No f-string; works even on older Python
    print("[DEBUG] recommend called with student_id=%s" % student_id)

    conn = get_db()
    cursor = conn.cursor()

    # 1) Get the tc_ids the student has already read
    cursor.execute("""
        SELECT DISTINCT tc_id
        FROM student_reads
        WHERE student_id = %s
    """, (student_id,))
    read_rows = cursor.fetchall()
    read_ids = [row["tc_id"] for row in read_rows]

    print("[DEBUG] student_reads rows for %s = %s" % (student_id, len(read_ids)))

    recs: List[Dict[str, Any]] = []

    # 2) If they have read something, recommend other theses
    if read_ids:
        placeholders = ",".join(["%s"] * len(read_ids))
        sql = """
            SELECT
                tc.tc_id,
                tc.title,
                tc.views,
                tc.colleges_id,
                tc.program_id,
                tc.abstract
            FROM thesis_capstone tc
            WHERE tc.is_archived = 0
              AND tc.tc_id NOT IN ({})
            ORDER BY tc.views DESC
            LIMIT 12
        """.format(placeholders)

        cursor.execute(sql, read_ids)
        recs = cursor.fetchall()

    # 3) If still no recs, fallback to trending (top viewed)
    if not recs:
        print("[DEBUG] no CF recs, falling back to trending")
        cursor.execute("""
            SELECT
                tc.tc_id,
                tc.title,
                tc.views,
                tc.colleges_id,
                tc.program_id,
                tc.abstract
            FROM thesis_capstone tc
            WHERE tc.is_archived = 0
            ORDER BY tc.views DESC
            LIMIT 12
        """)
        recs = cursor.fetchall()

    cursor.close()
    conn.close()

    print("[DEBUG] returning %s recommendations for student_id=%s" % (len(recs), student_id))
    return recs
