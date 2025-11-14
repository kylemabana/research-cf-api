import mysql.connector
from fastapi import FastAPI
from typing import List, Dict

app = FastAPI()

def get_db():
    return mysql.connector.connect(
        host="YOUR_DB_HOST",
        user="YOUR_DB_USER",
        password="YOUR_DB_PASS",
        database="u311577524_research_db"
    )

@app.get("/recommend")
def recommend(student_id: int) -> List[Dict]:
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    # 1) Get student's program + college
    cursor.execute("""
        SELECT program_id, colleges_id
        FROM student_information
        WHERE student_id = %s
        LIMIT 1
    """, (student_id,))
    student = cursor.fetchone()

    if not student:
        # If student doesn't exist → just return trending
        return get_trending(cursor)

    program_id = student["program_id"]
    colleges_id = student["colleges_id"]

    # 2) Get tc_ids the student already read
    cursor.execute("""
        SELECT DISTINCT tc_id
        FROM student_reads
        WHERE student_id = %s
    """, (student_id,))
    already_read_ids = [row["tc_id"] for row in cursor.fetchall()]

    # 3) Recommend theses in same program/college, not archived, not already read
    if already_read_ids:
        placeholders = ",".join(["%s"] * len(already_read_ids))
        cursor.execute(f"""
            SELECT tc.tc_id, tc.title, tc.views, tc.colleges_id, tc.program_id
            FROM thesis_capstone tc
            WHERE tc.is_archived = 0
              AND tc.program_id = %s
              AND tc.colleges_id = %s
              AND tc.tc_id NOT IN ({placeholders})
            ORDER BY tc.views DESC
            LIMIT 12
        """, (program_id, colleges_id, *already_read_ids))
    else:
        cursor.execute("""
            SELECT tc.tc_id, tc.title, tc.views, tc.colleges_id, tc.program_id
            FROM thesis_capstone tc
            WHERE tc.is_archived = 0
              AND tc.program_id = %s
              AND tc.colleges_id = %s
            ORDER BY tc.views DESC
            LIMIT 12
        """, (program_id, colleges_id))

    recs = cursor.fetchall()

    # 4) If still empty, backend fallback → trending
    if not recs:
        recs = get_trending(cursor)

    cursor.close()
    conn.close()
    return recs

def get_trending(cursor):
    cursor.execute("""
        SELECT tc.tc_id, tc.title, tc.views, tc.colleges_id, tc.program_id
        FROM thesis_capstone tc
        WHERE tc.is_archived = 0
        ORDER BY tc.views DESC
        LIMIT 12
    """)
    return cursor.fetchall()
