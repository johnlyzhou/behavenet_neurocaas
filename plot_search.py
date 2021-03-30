import numpy as np
from sklearn.cluster import KMeans
from behavenet.plotting.cond_ae_utils import (
    plot_psvae_training_curves,
    plot_label_reconstructions,
    make_latent_traversal_movie
)
from hyperparameter_search import (
    hyperparameter_search
)
from psvae_experiment import PSvaeExperiment
from search_utils import (
    load_latents_trials_frames,
    list_hparams
)


def _cluster(latents, n_clusters=5):
    # np.random.seed(0)  # to reproduce clusters
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    distances = kmeans.fit_transform(latents)
    clust_id = kmeans.predict(latents)
    return distances, clust_id


def plot_psvae_training_curves_wrapper(lab, expt, animal, session, expt_name, n_ae_latents, rng_seeds_model, n_labels,
                                       which_as_array='alphas', **kwargs):
    alphas, betas, gammas = list_hparams(lab, expt, animal, session, expt_name, n_ae_latents)
    n_ae_latents = [n_ae_latents - n_labels]
    if which_as_array == 'alphas':
        try:
            betas = betas[:1]
            gammas = gammas[:1]
            rng_seeds_model = rng_seeds_model[:1]
            n_ae_latents = n_ae_latents[:1]
        except TypeError:
            pass
    if which_as_array == 'betas':
        try:
            alphas = alphas[:1]
            gammas = gammas[:1]
            rng_seeds_model = rng_seeds_model[:1]
            n_ae_latents = n_ae_latents[:1]
        except TypeError:
            pass
    if which_as_array == 'gammas':
        try:
            betas = betas[:1]
            alphas = alphas[:1]
            rng_seeds_model = rng_seeds_model[:1]
            n_ae_latents = n_ae_latents[:1]
        except TypeError:
            pass
    if which_as_array == 'rng_seeds_model':
        try:
            betas = betas[:1]
            gammas = gammas[:1]
            alphas = alphas[:1]
            n_ae_latents = n_ae_latents[:1]
        except TypeError:
            pass
    if which_as_array == 'n_ae_latents':
        try:
            betas = betas[:1]
            gammas = gammas[:1]
            rng_seeds_model = rng_seeds_model[:1]
            alphas = alphas[:1]
        except TypeError:
            pass

    plot_psvae_training_curves(lab, expt, animal, session, alphas, betas, gammas, n_ae_latents, rng_seeds_model,
                               expt_name, n_labels, **kwargs)


def plot_label_reconstructions_wrapper(lab, expt, animal, session, n_ae_latents, experiment_name, n_labels, trials,
                                       alpha, beta, gamma, **kwargs):
    n_ae_latents -= n_labels
    plot_label_reconstructions(lab, expt, animal, session, n_ae_latents, experiment_name,
                               n_labels, trials, alpha=alpha, beta=beta, gamma=gamma, **kwargs)


def make_latent_traversal_movie_wrapper(lab, expt, animal, session, label_names, expt_name, n_ae_latents, alpha, beta,
                                        gamma, model_class='ps-vae', n_clusters=5, rng_seed_model=0, save_file=None,
                                        **kwargs):
    if save_file is None:
        save_file = '~/hello'

    movie_expt = PSvaeExperiment(lab, expt, animal, session, label_names, expt_name, n_ae_latents, alpha, beta, gamma,
                                 model_class=model_class, **kwargs)

    # need to test model_ae in load_latents_trials_frames
    latents, trials, frames = load_latents_trials_frames(movie_expt.hparams, movie_expt.version)
    traversal_trials = []
    batch_idxs = []
    distances, _ = _cluster(latents, n_clusters)
    for clust in range(n_clusters):
        frame_idx = np.argmin(distances[:, clust])
        traversal_trials.append(trials[frame_idx])
        batch_idxs.append(frames[frame_idx])
    trial_idxs = [None] * len(traversal_trials)
    make_latent_traversal_movie(lab, expt, animal, session, model_class, alpha, beta, gamma,
                                n_ae_latents - len(label_names), rng_seed_model, expt_name, len(label_names),
                                trial_idxs, batch_idxs, traversal_trials, save_file=save_file, **kwargs)


def plot_and_film_best(lab, expt, animal, session, label_names, expt_name, n_ae_latents, trials, beta=1, gamma=0,
                       model_class='ps-vae', n_clusters=5, rng_seed_model=0, rng_seeds_model=None, save_file=None,
                       **kwargs):

    if rng_seeds_model is None:
        rng_seeds_model = [rng_seed_model]
    alphas, betas, gammas = list_hparams(lab, expt, animal, session, expt_name, n_ae_latents)
    print(alphas, betas, gammas)
    alpha, beta, gamma = hyperparameter_search(lab, expt, animal, session, label_names, expt_name, n_ae_latents,
                                               alphas, betas, gammas, beta=beta, gamma=gamma, **kwargs)
    plot_psvae_training_curves_wrapper(lab, expt, animal, session, expt_name, n_ae_latents, rng_seeds_model,
                                       len(label_names), **kwargs)
    plot_label_reconstructions_wrapper(lab, expt, animal, session, n_ae_latents, expt_name, len(label_names), trials,
                                       alpha, beta, gamma, **kwargs)
    make_latent_traversal_movie_wrapper(lab, expt, animal, session, label_names, expt_name, n_ae_latents, alpha, beta,
                                        gamma, model_class=model_class, n_clusters=n_clusters,
                                        rng_seed_model=rng_seed_model, save_file=save_file, **kwargs)
