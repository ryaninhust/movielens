import dpark
import math
PRIOR_COUNT = 10
PRIOR_CORRELATION = 0
FILE_SUFFIX = 'ml-100k/'
TRAIN_FILENAME = 'ua.base'
TEST_FILENAME = 'ua.test'
MOVIES_FILENAME = 'u.item'


def _split_movie(line):
    fields = line.split('|')
    return int(fields[0]), fields[1]
movies = dpark.textFile(FILE_SUFFIX + MOVIES_FILENAME).map(_split_movie)
movie_names = movies.collectAsMap()
ratings = dpark.textFile(FILE_SUFFIX + TRAIN_FILENAME)


def _split_rating(line):
    fields = line.split('\t')
    return int(fields[0]), int(fields[1]), int(fields[2])
num_raters_perMovie = ratings.map(_split_rating).groupBy(lambda line: line[1])\
    .map(lambda line: (line[0], len(line[1])))
rating_with_size = ratings.map(_split_rating).groupBy(lambda line: line[1])\
    .join(num_raters_perMovie)


def _map_fields(line):
    return map(lambda f: (f[0], f[1], f[2], line[1][1]), line[1][0])

rating_with_size = rating_with_size.flatMap(_map_fields)
rating2 = rating_with_size.map(lambda line: (line[0], line))
rating_pairs = rating_with_size.map(lambda line: (line[0], line))\
    .join(rating2).filter(lambda f: f[1][0][1] > f[1][1][1])


def calcs(data):
    key = (data[1][0][1], data[1][1][1])
    stats = (data[1][0][2] * data[1][1][2],
             data[1][0][2],
             data[1][1][2],
             math.pow(data[1][0][2], 2),
             math.pow(data[1][1][2], 2),
             data[1][0][3],
             data[1][1][3],
             )
    return (key, stats)


def list_stat(data):
    key = data[0]
    vals = data[1]
    size = len(vals)
    dot_product = sum(map(lambda f: f[0], vals))
    rating_sum = sum(map(lambda f: f[1], vals))
    rating2_sum = sum(map(lambda f: f[2], vals))
    rating_sq = sum(map(lambda f: f[3], vals))
    rating2_sq = sum(map(lambda f: f[4], vals))
    num_raters = max(map(lambda f: f[5], vals))
    num_raters2 = max(map(lambda f: f[6], vals))
    return (key, (size, dot_product, rating_sum, rating2_sum, rating_sq,
            rating2_sq, num_raters, num_raters2))

vector_calcs = rating_pairs.map(calcs).groupByKey().map(list_stat)


def correlation(size, dot_product, rating_sum, rating2_sum, rating_norm_sq,
                rating2_norm_sq):
    numberator = size * dot_product - rating2_sum * rating_sum
    denominator = math.sqrt(size * rating_norm_sq - rating_sum * rating_sum)\
        * math.sqrt(size * rating2_norm_sq - rating2_sum * rating2_sum)
    return numberator / denominator


def regularizedCorrelation(size, dot_product, rating_sum, rating2_sum,
                           rating_norm_sq, rating2_norm_sq, virtual_count,
                           priorCorrelation):
    unregularized = correlation(size, dot_product, rating_sum, rating2_sum,
                                rating_norm_sq, rating2_norm_sq)
    w = float(size) / (size + virtual_count)
    return w * unregularized + (1-w) * priorCorrelation


def cosine_similarity(dot_product, rating_norm, rating2_norm):
    return dot_product / (rating2_norm * rating_norm)


def jaccard_similarity(user_in_common, total_user1, total_user2):
    union = total_user1 + total_user2 - user_in_common
    return float(user_in_common) / union


def calc_similaritiy(fields):
    key = fields[0]
    args = fields[1]
    corr = correlation(*args[:6])
    reg_corr = regularizedCorrelation(*args[:6], virtual_count=PRIOR_COUNT,
                                      priorCorrelation=PRIOR_CORRELATION)
    cos_sim = cosine_similarity(args[
                                1], math.sqrt(args[-4]), math.sqrt(args[-3]))
    jaccard = jaccard_similarity(args[0], args[-2], args[-1])
    return (key, (corr, reg_corr, cos_sim, jaccard))

similarities = vector_calcs.map(calc_similaritiy)
print similarities.take(1)
