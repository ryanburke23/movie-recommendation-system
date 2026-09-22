import numpy as np


def train_matrix_factorization(
    train,
    user_to_idx,
    movie_to_idx,
    num_users,
    num_movies,
    n_factors=10,
    learning_rate=0.005,
    regularization=0.02,
    epochs=20,
    seed=42
):
    rng = np.random.default_rng(seed)

    P = rng.normal(
        0,
        0.1,
        size=(num_users, n_factors)
    )

    Q = rng.normal(
        0,
        0.1,
        size=(num_movies, n_factors)
    )

    user_bias = np.zeros(num_users)
    movie_bias = np.zeros(num_movies)

    global_mean = train["rating"].mean()

    training_rmse = []

    train_array = train[
        ["user_id", "movie_id", "rating"]
    ].to_numpy()

    for epoch in range(epochs):

        rng.shuffle(train_array)

        squared_error = 0

        for user_id, movie_id, rating in train_array:

            u = user_to_idx[user_id]
            i = movie_to_idx[movie_id]

            prediction = (
                global_mean
                + user_bias[u]
                + movie_bias[i]
                + np.dot(P[u], Q[i])
            )

            error = rating - prediction
            squared_error += error ** 2

            p_old = P[u].copy()

            user_bias[u] += learning_rate * (
                error - regularization * user_bias[u]
            )

            movie_bias[i] += learning_rate * (
                error - regularization * movie_bias[i]
            )

            P[u] += learning_rate * (
                error * Q[i]
                - regularization * P[u]
            )

            Q[i] += learning_rate * (
                error * p_old
                - regularization * Q[i]
            )

        rmse = np.sqrt(
            squared_error / len(train_array)
        )

        training_rmse.append(rmse)

        print(
            f"Epoch {epoch + 1:2d}/{epochs} "
            f"- Training RMSE: {rmse:.4f}"
        )

    return (
        P,
        Q,
        user_bias,
        movie_bias,
        global_mean,
        training_rmse
    )