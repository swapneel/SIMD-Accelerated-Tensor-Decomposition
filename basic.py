import math
import numpy as np
import timeit

def cosine_distance(a, b):
    dot_product = sum(ai * bi for ai, bi in zip(a, b))
    magnitude_a = math.sqrt(sum(ai * ai for ai in a))
    magnitude_b = math.sqrt(sum(bi * bi for bi in b))
    cosine_similarity = dot_product / (magnitude_a * magnitude_b)
    return 1 - cosine_similarity

def cosine_distance_numpy(a, b):
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    cosine_similarity = dot_product / (magnitude_a * magnitude_b)
    return 1 - cosine_similarity

np.random.seed(0)
a_list = np.random.rand(1536).tolist()
b_list = np.random.rand(1536).tolist()

# Measure the performance
execution_time = timeit.timeit('cosine_distance(a_list, b_list)', globals=globals(), number=1000)
average_execution_time_ms = (execution_time / 1000) * 1000  # Convert seconds to milliseconds
print("Average execution time regular: {:.2f} ms".format(average_execution_time_ms))

execution_time = timeit.timeit('cosine_distance_numpy(a_list, b_list)', globals=globals(), number=1000)
average_execution_time_ms = (execution_time / 1000) * 1000  # Convert seconds to milliseconds
print("Average execution time regular: {:.2f} ms".format(average_execution_time_ms))

# Measure the performance with optimized data structures
print("tests with optimized numpy types")
a_list = np.random.rand(1536).astype(np.int8)
b_list = np.random.rand(1536).astype(np.int8)
execution_time = timeit.timeit('cosine_distance(a_list, b_list)', globals=globals(), number=1000)
average_execution_time_ms = (execution_time / 1000) * 1000  # Convert seconds to milliseconds
print("Average execution time regular: {:.2f} ms".format(average_execution_time_ms))

execution_time = timeit.timeit('cosine_distance_numpy(a_list, b_list)', globals=globals(), number=1000)
average_execution_time_ms = (execution_time / 1000) * 1000  # Convert seconds to milliseconds
print("Average execution time regular: {:.2f} ms".format(average_execution_time_ms))
