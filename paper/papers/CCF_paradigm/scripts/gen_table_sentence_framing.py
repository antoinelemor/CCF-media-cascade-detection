#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-media-cascade-detection

TITLE:
------
gen_table_sentence_framing.py

MAIN OBJECTIVE:
---------------
Compute sentence-level frame proportions for specific event contexts,
demonstrating how elections absorb economic framing into politics and
how scientific publications generate multi-frame debates.

Usage:
  python paper/papers/CCF_paradigm/scripts/gen_table_sentence_framing.py

Output:
  tables/table_sentence_framing.tex
  tables/sentence_framing.csv

Author:
-------
Antoine Lemor
"""
import os
import psycopg2
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
OUT = Path(__file__).resolve().parent.parent / 'tables'
load_dotenv(ROOT / '.env')

FRAMES_DB = ['political_frame', 'economic_frame', 'scientific_frame',
             'environmental_frame']
FRAMES_SHORT = ['Pol', 'Eco', 'Sci', 'Env']


def query(cur, sql):
    cur.execute(sql)
    return cur.fetchone()


def main():
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        dbname='CCF_Database_texts'
    )
    cur = conn.cursor()

    results = []

    # 1. 2008 election: carbon tax sentences during campaign
    r = query(cur, """
        SELECT COUNT(*),
               SUM(political_frame::int), SUM(economic_frame::int),
               SUM(scientific_frame::int), SUM(environmental_frame::int)
        FROM "CCF_processed_data"
        WHERE date BETWEEN '2008-09-01' AND '2008-10-15'
        AND evt_election > 0.3
        AND (sentences ILIKE '%%carbon tax%%' OR sentences ILIKE '%%green shift%%'
             OR sentences ILIKE '%%cap and trade%%' OR sentences ILIKE '%%carbon price%%')
    """)
    n = r[0]
    results.append({
        'Context': '2008 election, carbon tax sentences',
        'n': n,
        'Pol': round(100 * r[1] / n, 0),
        'Eco': round(100 * r[2] / n, 0),
        'Sci': round(100 * r[3] / n, 0),
        'Env': round(100 * r[4] / n, 0),
    })

    # 2. 2008 non-election: carbon tax sentences
    r = query(cur, """
        SELECT COUNT(*),
               SUM(political_frame::int), SUM(economic_frame::int),
               SUM(scientific_frame::int), SUM(environmental_frame::int)
        FROM "CCF_processed_data"
        WHERE date BETWEEN '2008-01-01' AND '2008-08-31'
        AND (sentences ILIKE '%%carbon tax%%' OR sentences ILIKE '%%green shift%%'
             OR sentences ILIKE '%%cap and trade%%' OR sentences ILIKE '%%carbon price%%')
    """)
    n = r[0]
    results.append({
        'Context': '2008 non-election, carbon tax sentences',
        'n': n,
        'Pol': round(100 * r[1] / n, 0),
        'Eco': round(100 * r[2] / n, 0),
        'Sci': round(100 * r[3] / n, 0),
        'Env': round(100 * r[4] / n, 0),
    })

    # 3. IPCC SR15 release (Oct 8-14, 2018)
    r = query(cur, """
        SELECT COUNT(*),
               SUM(political_frame::int), SUM(economic_frame::int),
               SUM(scientific_frame::int), SUM(environmental_frame::int)
        FROM "CCF_processed_data"
        WHERE date BETWEEN '2018-10-08' AND '2018-10-14'
        AND evt_publication > 0.3
    """)
    n = r[0]
    results.append({
        'Context': 'IPCC SR15 release (Oct 8--14, 2018)',
        'n': n,
        'Pol': round(100 * r[1] / n, 0),
        'Eco': round(100 * r[2] / n, 0),
        'Sci': round(100 * r[3] / n, 0),
        'Env': round(100 * r[4] / n, 0),
    })

    # 4. Baseline: all publication sentences in 2018
    r = query(cur, """
        SELECT COUNT(*),
               SUM(political_frame::int), SUM(economic_frame::int),
               SUM(scientific_frame::int), SUM(environmental_frame::int)
        FROM "CCF_processed_data"
        WHERE date BETWEEN '2018-01-01' AND '2018-12-31'
        AND evt_publication > 0.3
    """)
    n = r[0]
    results.append({
        'Context': 'All publication sentences, 2018 (baseline)',
        'n': n,
        'Pol': round(100 * r[1] / n, 0),
        'Eco': round(100 * r[2] / n, 0),
        'Sci': round(100 * r[3] / n, 0),
        'Env': round(100 * r[4] / n, 0),
    })

    conn.close()
    rdf = pd.DataFrame(results)

    print("Sentence-level framing by context:")
    print(rdf.to_string(index=False))

    # LaTeX table
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\renewcommand{\arraystretch}{1.15}',
        r'\caption{\textbf{Sentence-level frame proportions in specific event contexts.}',
        r'Each row reports the percentage of sentences classified under each frame.',
        r'The 2008 election absorbed economic framing into politics',
        r'(Eco dropped from 39\% to 10\%). The IPCC SR15 release amplified',
        r'political framing (+14 pp) more than scientific framing (+10 pp)',
        r'relative to the 2018 publication baseline.}',
        r'\label{tab:si_sentence_framing}',
        r'\vspace{0.3em}',
        r'\small',
        r'\begin{tabular}{@{}l r r r r r@{}}',
        r'\toprule',
        r'Context & $n$ & Pol (\%) & Eco (\%) & Sci (\%) & Env (\%) \\',
        r'\midrule',
    ]
    for _, r in rdf.iterrows():
        lines.append(
            f"{r['Context']} & {int(r['n']):,} & {int(r['Pol'])} & "
            f"{int(r['Eco'])} & {int(r['Sci'])} & {int(r['Env'])} \\\\"
        )
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    OUT.mkdir(exist_ok=True)
    tex_path = OUT / 'table_sentence_framing.tex'
    tex_path.write_text('\n'.join(lines))
    print(f"\nSaved: {tex_path}")

    csv_path = OUT / 'sentence_framing.csv'
    rdf.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == '__main__':
    main()
