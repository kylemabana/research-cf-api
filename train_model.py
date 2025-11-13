
import pymysql

import os
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from extract_pdf_text import extract_pdf_text

# MySQL connection config
DB_CONFIG = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '',
    'database': 'research_db',
    'cursorclass': pymysql.cursors.DictCursor
}

def main():
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            cursor.execute('''
                SELECT atv.tc_id, atv.title, atv.authorone, atv.authortwo, tc.file
                FROM approved_thesis_view atv
                LEFT JOIN thesis_capstone tc ON atv.tc_id = tc.tc_id
            ''')
            rows = cursor.fetchall()
    finally:
        conn.close()

    data = []
    for row in rows:
        authors = ', '.join([row.get('authorone') or '', row.get('authortwo') or '']).strip(', ').replace(',,', ',')
        pdf_path = row.get('file', '')
        if pdf_path and not os.path.isabs(pdf_path):
            if not pdf_path.startswith('uploads'):
                pdf_path = os.path.join('uploads', pdf_path)
            pdf_path = os.path.abspath(pdf_path)
        pdf_text = ''
        if pdf_path:
            print(f"[TRAIN] Indexing PDF: {pdf_path} Exists: {os.path.isfile(pdf_path)}")
        if pdf_path and os.path.isfile(pdf_path):
            pdf_text = extract_pdf_text(pdf_path)
            if not pdf_text.strip():
                print(f"[TRAIN] PDF {pdf_path} has NO extractable text.")
            else:
                print(f"[TRAIN] PDF {pdf_path} first 300 chars: {pdf_text[:300]}")
        elif pdf_path:
            print(f"[TRAIN] PDF {pdf_path} is missing!")
        full_text = ' '.join([row.get('title') or '', authors, pdf_text])
        data.append(full_text)
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(data)
    joblib.dump(vectorizer, 'vectorizer.pkl')
    joblib.dump(tfidf_matrix, 'tfidf_matrix.pkl')
    print('TF-IDF model and matrix saved.')

if __name__ == '__main__':
    main()
