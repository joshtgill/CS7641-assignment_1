import numpy as np
from datetime import datetime


'''
A standard str.split(',') or regex split doesn't work for movies_metadata.csv due
to the frequent and nested commas, single quotes, double quotes, brackets, and braces.

A couple hundred movies are still lost to formatting.
'''
def preprocess(annoying_csv):
    rows = []
    with open(annoying_csv, 'r') as file:
        lines = file.readlines()
        num_cols = len(lines[0].split(','))
        for line in lines[1 :]:
            line = line.strip()

            data = ['']
            double_quote_iter = single_quote_iter = bracket_iter = brace_iter = 0
            for c in line:
                if c == ',' and double_quote_iter == single_quote_iter == bracket_iter == brace_iter == 0:
                    data.append('')
                    continue

                if c == '"' and not double_quote_iter:
                    # Opened a double quote
                    double_quote_iter += 1
                elif c == '"' and double_quote_iter:
                    # Closed a double quote
                    double_quote_iter -= 1
                elif c == "'" and (bracket_iter or brace_iter) and not single_quote_iter and not double_quote_iter:
                    # Opened a single quote while in a bracket/brace, but not in a double quote
                    single_quote_iter += 1
                elif c == "'" and (bracket_iter or brace_iter) and single_quote_iter:
                    # Closed a single quote while in a backet/brace
                    single_quote_iter -= 1
                    # Opened a bracket
                elif c == '[':
                    bracket_iter += 1
                    # Closed a bracket
                elif c == ']':
                    bracket_iter -= 1
                    # Opened a brace
                elif c == '{':
                    brace_iter += 1
                    # Closed a brace
                elif c == '}':
                    brace_iter -= 1

                # Add character to active index
                data[-1] += c

            if len(data) == num_cols:
                rows.append(data)

    return rows


POP_THRESHOLD = 18 # popularity mean=9.778863736607143 stddev=8.392760500000001

def filter(rows):

    data = []
    for movie in rows:
        try:
            title = movie[8]
            runtime = float(movie[16])
            release_year = datetime.strptime(movie[14], '%Y-%m-%d').year
            isFranchise = int(1 if movie[1] else 0)
            budget = int(movie[2])
            revenue = int(movie[15])
            popularity = int(float(movie[10]) > POP_THRESHOLD)
            score = float(movie[22])

            if budget == 0 or revenue == 0:
                continue

            data.append([runtime, release_year, isFranchise, budget, revenue, popularity])
        except ValueError:
            continue

    data = np.array(data)
    np.random.shuffle(data)
    return data


def extract_data():
    return filter(preprocess('data/movies_metadata.csv'))
