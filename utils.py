
from collections import defaultdict
import pandas as pd

SEQUENTIAL_HERO_ID_TO_ACTUAL_HERO_ID = {100: 102, 84: 86, 85: 87, 97: 99, 122: 136, 1: 2, 74: 76, 23: 25, 116: 123, 107: 109, 47: 49, 52: 54, 67: 69, 8: 9, 124: 138, 60: 62, 73: 75, 59: 61, 94: 96, 16: 17, 51: 53, 30: 32, 34: 36, 4: 5, 9: 10, 98: 100, 29: 31, 44: 46, 108: 110, 18: 19, 75: 77, 33: 35, 24: 26, 42: 44, 89: 91, 50: 52, 70: 72, 35: 37, 15: 16, 114: 120, 43: 45, 109: 111, 41: 43, 12: 13, 46: 48, 83: 85, 61: 63, 20: 21, 26: 28, 27: 29, 6: 7, 25: 27, 48: 50, 10: 11, 14: 15, 110: 112, 49: 51, 57: 59, 62: 64, 112: 114, 63: 65, 121: 135, 87: 89, 37: 39, 53: 55, 101: 103, 123: 137, 13: 14, 32: 34, 21: 22, 36: 38, 120: 131, 104: 106, 82: 84, 103: 105, 102: 104, 91: 93, 58: 60, 79: 81, 31: 33, 17: 18, 99: 101, 95: 97, 22: 23, 0: 1, 115: 121, 40: 42, 117: 126, 38: 40, 119: 129, 3: 4, 56: 58, 90: 92, 69: 71, 106: 108, 19: 20, 39: 41, 28: 30, 93: 95, 113: 119, 7: 8, 68: 70, 65: 67, 45: 47, 72: 74, 11: 12, 5: 6, 2: 3, 66: 68, 118: 128, 96: 98, 77: 79, 71: 73, 88: 90, 105: 107, 76: 78, 111: 113, 86: 88, 92: 94, 80: 82, 78: 80, 81: 83, 64: 66, 54: 56, 55: 57}
ACTUAL_HERO_ID_TO_HERO_NAME = {102: 'Abaddon', 86: 'Rubick', 87: 'Disruptor', 99: 'Bristleback', 136: 'Marci', 2: 'Axe', 76: 'Outworld Destroyer', 25: 'Lina', 123: 'Hoodwink', 109: 'Terrorblade', 49: 'Dragon Knight', 54: 'Lifestealer', 69: 'Doom', 9: 'Mirana', 138: 'Muerta', 62: 'Bounty Hunter', 75: 'Silencer', 61: 'Broodmother', 96: 'Centaur Warrunner', 17: 'Storm Spirit', 53: "Nature's Prophet", 32: 'Riki', 36: 'Necrophos', 5: 'Crystal Maiden', 10: 'Morphling', 100: 'Tusk', 31: 'Lich', 46: 'Templar Assassin', 110: 'Phoenix', 19: 'Tiny', 77: 'Lycan', 35: 'Sniper', 26: 'Lion', 44: 'Phantom Assassin', 91: 'Io', 52: 'Leshrac', 72: 'Gyrocopter', 37: 'Warlock', 16: 'Sand King', 120: 'Pangolier', 45: 'Pugna', 111: 'Oracle', 43: 'Death Prophet', 13: 'Puck', 48: 'Luna', 85: 'Undying', 63: 'Weaver', 21: 'Windranger', 28: 'Slardar', 29: 'Tidehunter', 7: 'Earthshaker', 27: 'Shadow Shaman', 50: 'Dazzle', 11: 'Shadow Fiend', 15: 'Razor', 112: 'Winter Wyvern', 51: 'Clockwerk', 59: 'Huskar', 64: 'Jakiro', 114: 'Monkey King', 65: 'Batrider', 135: 'Dawnbreaker', 89: 'Naga Siren', 39: 'Queen of Pain', 55: 'Dark Seer', 103: 'Elder Titan', 137: 'Primal Beast', 14: 'Pudge', 34: 'Tinker', 22: 'Zeus', 38: 'Beastmaster', 131: 'Ringmaster', 106: 'Ember Spirit', 84: 'Ogre Magi', 105: 'Techies', 104: 'Legion Commander', 93: 'Slark', 60: 'Night Stalker', 81: 'Chaos Knight', 33: 'Enigma', 18: 'Sven', 101: 'Skywrath Mage', 97: 'Magnus', 23: 'Kunkka', 1: 'Anti-Mage', 121: 'Grimstroke', 42: 'Wraith King', 126: 'Void Spirit', 40: 'Venomancer', 129: 'Mars', 4: 'Bloodseeker', 58: 'Enchantress', 92: 'Visage', 71: 'Spirit Breaker', 108: 'Underlord', 20: 'Vengeful Spirit', 41: 'Faceless Void', 30: 'Witch Doctor', 95: 'Troll Warlord', 119: 'Dark Willow', 8: 'Juggernaut', 70: 'Ursa', 67: 'Spectre', 47: 'Viper', 74: 'Invoker', 12: 'Phantom Lancer', 6: 'Drow Ranger', 3: 'Bane', 68: 'Ancient Apparition', 128: 'Snapfire', 98: 'Timbersaw', 79: 'Shadow Demon', 73: 'Alchemist', 90: 'Keeper of the Light', 107: 'Earth Spirit', 78: 'Brewmaster', 113: 'Arc Warden', 88: 'Nyx Assassin', 94: 'Medusa', 82: 'Meepo', 80: 'Lone Druid', 83: 'Treant Protector', 66: 'Chen', 56: 'Clinkz', 57: 'Omniknight'}

def seq_hero_id_2_hero_id(seq_hero_id):
    return SEQUENTIAL_HERO_ID_TO_ACTUAL_HERO_ID[seq_hero_id]


def hero_id_2_hero_name(hero_id):
    return ACTUAL_HERO_ID_TO_HERO_NAME[hero_id]


def evaluate_state_evaluator(df, scoring_function, hero_col="sequential_hero_id", order_col="move_order"):
    """
    Evaluates the scoring_function at every draft state from the dataframe.

    :param df: pandas Dataframe with the columns ['match_id', 'team', hero_col, order_col, 'winner', 'choice']
    :param scoring_function: partial function of the scoring function we're evaluating. 
    :param hero_col: column we use for the hero id, defaults to "sequential_hero_id"
    :param order_col: column we use for the draft order, defaults to "move_order"
    :return: dictionary of metrics
    """
    picks = (
        df[df["choice"] == "pick"]
        .sort_values(["match_id", order_col])
    )

    stage_correct = defaultdict(int)
    stage_total   = defaultdict(int)

    for match_id, g in picks.groupby("match_id", sort=False):
        actual = g["winner"].iloc[0]         # 'radiant' or 'dire'
        radiant, dire = [], []

        for _, row in g.iterrows():
            (radiant if row["team"] == "radiant" else dire).append(row[hero_col])

            stage = len(radiant) + len(dire)     # 1‑based stage index
            if stage % 2:                      # skip odd stages 1,3,5,7,9
                continue
            
            pred = scoring_function(radiant, dire)
            stage_correct[stage] += (pred == actual)
            stage_total[stage]   += 1

    stage_acc = {s: stage_correct[s] / stage_total[s]
                 for s in sorted(stage_total)}

    overall = sum(stage_correct.values()) / sum(stage_total.values())

    return {"stage_accuracy": stage_acc, "overall_accuracy": overall}