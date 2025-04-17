import abc
from collections import defaultdict
from pathlib import Path
import pickle
from itertools import combinations, product
import os


class DraftStateEvaluator:
    def __init__(self, df):
        """
        Creates a draft state evaluator function

        :param df: the dataframe for the underlying historical game data we're using to fit
        """
        self.data = df
        self._cache_location = self._init_cache() 
    
    def _init_cache(self):
        cache_location = Path(os.getenv("HOME"), ".iron-talon")
        cache_location.mkdir(parents=True, exist_ok=True)
        return cache_location
    
    def get_or_compute(self, name, compute_fn):
        """
        If `name` exists on disk, load it. Otherwise, compute it via `compute_fn`,
        save it, and return the computed result.
        """
        path = self._cache_location / name
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        # If we don't have the data cached, compute it and save.
        result = compute_fn()
        with open(path, "wb") as f:
            pickle.dump(result, f)
        return result
        
    @abc.abstractmethod
    def evaluate(self, radiant_heroes: list[int], dire_heroes: list[int]) -> float:
        """
        Given the sequential hero_ids of the radiant and dire, return a "quality"
        score that represents how likely the radiant is to win

        :param radiant_heroes: a list of sequential hero ids for the radiant heroes
        :param dire_heroes: a list of sequential hero ids for the dire heroes
        :return: A float that represents the quality of the draft for the radiant
        """
        raise NotImplementedError("Implement evaluate!")
    
    def predict_winner(self, radiant_heroes: list[int], dire_heroes: list[int]):
        """
        Binary winner prediction given the evalute function

        :param radiant_heroes: a list of sequential hero ids for the radiant heroes
        :param dire_heroes: a list of sequential hero ids for the dire heroes
        :return: A string representing who will win, radiant or dire
        """
        score = self.evaluate(radiant_heroes, dire_heroes)
        if score >= 0.0:
            return 'radiant'
        else:
            return 'dire'
    

class NaiveHeroWinrateDraftStateEvaluator(DraftStateEvaluator):
    """
    Dumb baseline that simply looks to pick the hero with the best winrate
    """
    def __init__(self, df):
        super().__init__(df)
        self.winrates = self.get_or_compute("winrates", self._compute_hero_winrates)

    def _compute_hero_winrates(self):
        picks_df = self.df[self.df["choice"] == "pick"].copy()
        picks_df["is_win"] = (picks_df["team"] == picks_df["winner"]).astype(int)
        hero_stats = picks_df.groupby("sequential_hero_id").agg(
            picks=("sequential_hero_id", "count"),
            wins=("is_win", "sum")
        )
        hero_stats["winrate"] = hero_stats["wins"] / hero_stats["picks"]
        hero_winrate_dict = hero_stats["winrate"].to_dict()
        return hero_winrate_dict


    def evaluate(self, radiant_heroes, dire_heroes):
        radiant_score = sum(self.winrates.get(h) for h in radiant_heroes)
        dire_score = sum(self.winrates.get(h) for h in dire_heroes)
        return radiant_score - dire_score
    
    
class SynergyDraftStateEvaluator(DraftStateEvaluator):
    def __init__(self, df, default_synergy=0.1, alpha=1):
        super().__init__(df)
        self.synergy = self.get_or_compute("synergy", self._compute_pairwise_synergy)
        self.default_synergy = default_synergy
        self.alpha = alpha

    def _compute_pairwise_synergy(self):
        picks_df = self.df[self.df["choice"] == "pick"].copy()
        # stored as (heroA, heroB) = [appearances, wins]
        synergy_counts = defaultdict(lambda: [0, 0])  
        match_groups = picks_df.groupby("match_id")

        for match_id, group in match_groups:
            winners = group["winner"].unique()
            if len(winners) != 1:
                continue
            actual_winner = winners[0]
            radiant_heroes = group.loc[group["team"] == "radiant", 'sequential_hero_id'].tolist()
            dire_heroes = group.loc[group["team"] == "dire", 'sequential_hero_id'].tolist()

            # For Radiant, generate all pairs
            for (a, b) in combinations(radiant_heroes, 2):
                key = tuple(sorted((a, b)))
                synergy_counts[key][0] += 1  
                if actual_winner == "radiant":
                    synergy_counts[key][1] += 1

            # For Dire, generate all pairs
            for (a, b) in combinations(dire_heroes, 2):
                key = tuple(sorted((a, b)))
                synergy_counts[key][0] += 1
                if actual_winner == "dire":
                    synergy_counts[key][1] += 1
        
        synergy_dict = {}
        for pair, (appearances, wins) in synergy_counts.items():
            # We also add laplace smoothing for rare combos
            synergy_dict[pair] = (wins + self.alpha) / (appearances + 2 * self.alpha)  
        
        return synergy_dict


    def evaluate(self, radiant_heroes, dire_heroes):
        radiant_synergy = 0.0
        for (a, b) in combinations(radiant_heroes, 2):
            key = tuple(sorted((a, b)))
            radiant_synergy += self.synergy.get(key, self.default_synergy)  

        dire_synergy = 0.0
        for (a, b) in combinations(dire_heroes, 2):
            key = tuple(sorted((a, b)))
            dire_synergy += self.synergy.get(key, self.default_synergy)

        return radiant_synergy - dire_synergy

    
class CounterSynergyDraftStateEvaluator(SynergyDraftStateEvaluator):
    def __init__(self, df, default_synergy=0.1, alpha=1, default_counter=0.0):
        super().__init__(df, default_synergy=default_synergy, alpha=alpha)
        self.counter = self.get_or_compute("counter", self._compute_counter)
        self.default_counter = default_counter

    def _compute_counter(self):
        picks_df = self.df[self.df["choice"] == "pick"].copy()
        cross_counts = defaultdict(lambda: [0, 0])
        match_groups = picks_df.groupby("match_id")

        for match_id, group in match_groups:
            # Identify winner
            winners = group["winner"].unique()
            if len(winners) != 1:
                # data issue, skip
                continue
            actual_winner = winners[0]  # "radiant" or "dire"
            radiant_heroes = group.loc[group["team"] == "radiant", 'sequential_hero_id'].tolist()
            dire_heroes = group.loc[group["team"] == "dire", 'sequential_hero_id'].tolist()

            for (r, d) in product(radiant_heroes, dire_heroes):
                cross_counts[(r, d)][0] += 1  # appearances
                if actual_winner == "radiant":
                    cross_counts[(r, d)][1] += 1

        cross_dict = {}
        for pair, (appearances, rad_wins) in cross_counts.items():
            cross_dict[pair] = (rad_wins + self.alpha) / (appearances + 2 * self.alpha) 
        
        return cross_dict

    def evaluate(self, radiant_heroes, dire_heroes):
        radiant_synergy = 0.0
        for (a, b) in combinations(radiant_heroes, 2):
            key = tuple(sorted((a, b)))
            radiant_synergy += self.synergy.get(key, self.default_synergy)  

        dire_synergy = 0.0
        for (a, b) in combinations(dire_heroes, 2):
            key = tuple(sorted((a, b)))
            dire_synergy += self.synergy.get(key, self.default_synergy)

        radiant_counter_adv = 0.0
        for (r, d) in product(radiant_heroes, dire_heroes):
            radiant_cross_adv += self.counter.get((r, d), self.default_counter)

        dire_counter_adv = 0.0
        for (r, d) in product(radiant_heroes, dire_heroes):
            # cross_dict[(r, d)] => fraction Radiant wins
            rad_win_prob = self.counter.get((r, d), self.default_counter)
            # to keep it simple since we built counters from the radiant perspective,
            # the dire win prob is just 1 - the radiant win prob. 
            dire_win_prob = 1.0 - rad_win_prob  
            dire_counter_adv += dire_win_prob

        return (radiant_synergy + radiant_counter_adv) - (dire_synergy + dire_counter_adv)

    