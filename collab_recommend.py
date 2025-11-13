import pymysql
import pandas as pd
from collections import defaultdict

# MySQL connection config
DB_CONFIG = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '',
    'database': 'research_db',
    'cursorclass': pymysql.cursors.DictCursor
}

def get_user_thesis_matrix():
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            cursor.execute('SELECT student_id, tc_id FROM student_reads')
            rows = cursor.fetchall()
    finally:
        conn.close()
    # Build user-item matrix (binary: 1 if viewed)
    data = defaultdict(dict)
    for row in rows:
        data[row['student_id']][row['tc_id']] = 1
    df = pd.DataFrame(data).fillna(0).T  # users as rows, tc_id as columns
    return df

def recommend_for_user(target_student_id, top_n=5, k_neighbors=10, min_sim=0.0):
    matrix = get_user_thesis_matrix()
    # normalize index to str; normalize key to str
    matrix.index = matrix.index.astype(str)
    key = str(target_student_id)
    if key not in matrix.index:
        return []

    from sklearn.metrics import jaccard_score
    import numpy as np

    target_vec = matrix.loc[key].values
    sims = []
    for user in matrix.index:
        if user == key: 
            continue
        sim = jaccard_score(target_vec, matrix.loc[user].values, zero_division=0)
        if sim >= min_sim:
            sims.append((user, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    top_users = [u for u, _ in sims[:k_neighbors]]

    already = set(matrix.columns[target_vec > 0])
    votes = defaultdict(int)
    for u in top_users:
        for tc in matrix.columns[matrix.loc[u].values > 0]:
            if tc not in already:
                votes[tc] += 1

    if not votes:
        return []

    ranked = sorted(votes, key=votes.get, reverse=True)[:top_n]
    return ranked

def get_titles_by_tc_ids(tc_ids):
    if not tc_ids:
        return []
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            format_strings = ','.join(['%s'] * len(tc_ids))
            cursor.execute(f"SELECT tc_id, title FROM thesis_capstone WHERE tc_id IN ({format_strings})", tuple(tc_ids))
            rows = cursor.fetchall()
    finally:
        conn.close()
    # Return titles sorted by input tc_id order
    id_to_title = {row['tc_id']: row['title'] for row in rows}
    return [id_to_title.get(tc_id, str(tc_id)) for tc_id in tc_ids]

def recommend_by_college_program(college_id, program_id, top_n=5):
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            sql = "SELECT tc_id, title FROM thesis_capstone WHERE 1"
            params = []
            if college_id:
                sql += " AND colleges_id = %s"
                params.append(college_id)
            if program_id:
                sql += " AND program_id = %s"
                params.append(program_id)
            sql += " ORDER BY views DESC LIMIT %s"
            params.append(top_n)
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
    finally:
        conn.close()
    return [row['title'] for row in rows]

def recommend_most_visited(top_n=5):
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT title, views FROM thesis_capstone ORDER BY views DESC, tc_id DESC LIMIT %s", (top_n,))
            rows = cursor.fetchall()
            # If all views are zero, fallback to latest theses
            if not rows or all((row['views'] or 0) == 0 for row in rows):
                cursor.execute("SELECT title FROM thesis_capstone ORDER BY tc_id DESC LIMIT %s", (top_n,))
                rows = cursor.fetchall()
    finally:
        conn.close()
    return [row['title'] for row in rows]

if __name__ == '__main__':
    import sys
    import json
    if len(sys.argv) < 2:
        print(json.dumps([]))
        sys.exit(0)
    mode = sys.argv[1].lower()
    if mode == 'recommend' and len(sys.argv) >= 3:
        student_id = int(sys.argv[2])
        rec_ids = recommend_for_user(student_id)
        titles = get_titles_by_tc_ids(rec_ids)
        print(json.dumps(titles, ensure_ascii=False))
    elif mode == 'college' and len(sys.argv) >= 4:
        college_id = sys.argv[2]
        program_id = sys.argv[3]
        titles = recommend_by_college_program(college_id, program_id)
        print(json.dumps(titles, ensure_ascii=False))
    elif mode == 'views':
        titles = recommend_most_visited()
        print(json.dumps(titles, ensure_ascii=False))
    else:
        print(json.dumps([]))
