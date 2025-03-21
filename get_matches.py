import argparse

import opendota
import time
import sqlite3
from tqdm import tqdm


def main(args):
    client = opendota.OpenDota(api_key=args.key)

    conn = sqlite3.connect('dota_matches.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS match_moves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            choice TEXT,
            hero_id INTEGER,
            team TEXT,
            move_order INTEGER,
            match_id INTEGER,
            winner TEXT
        )
    ''')
    conn.commit()

    master_max_retries = 2500
    current_retries_cnt = 0
    try:
        current_min_matchid = args.match_id
        for _ in (pbar := tqdm(range(650))):
            match_infos = client.get_pro_matches(match_id=current_min_matchid)
            pbar.set_description(f"Min match id: {current_min_matchid}")
            time.sleep(0.2)
            for mi in (lower_pbar := tqdm(match_infos, leave=False)):
                id = mi['match_id']
                winner = 'radiant' if mi['radiant_win'] else 'dire'
                
                lower_pbar.set_description(f"Current Match ID: {id}")

                # Retry logic for client.get_match up to 5 times
                match = None
                for _ in range(5):
                    current_retries_cnt += 1
                    match = client.get_match(str(id))
                    if match is not None:
                        break
                    time.sleep(0.5) 

                # If still no match, continue to next iteration
                if match is None:
                    continue

                # not captains draft? 
                if "picks_bans" not in match:
                    continue

                draft_order = match["picks_bans"]
                if draft_order is None:
                    continue # ????

                for move in draft_order:
                    choice = "pick" if move["is_pick"] else "ban"
                    hero_id = move["hero_id"]
                    team = 'radiant' if move["team"] == 0 else 'dire'
                    order = move["order"]

                    cursor.execute('''
                        INSERT INTO match_moves (choice, hero_id, team, move_order, match_id, winner)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (choice, hero_id, team, order, id, winner))
                    conn.commit()
                    
                time.sleep(0.1)
            current_min_matchid = mi["match_id"]

        conn.close()
    finally:
        conn.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, help="OpenDota API key")
    parser.add_argument("--match_id", type=int, help="The match id to start from for getting matches")
    args = parser.parse_args()

    main(args)
