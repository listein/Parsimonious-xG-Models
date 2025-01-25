from auxiliaries import*
import os, json
import pandas as pd

matches_path = '/Users/livio/Documents/Bachelor Thesis/python/xG Analysis/open-data/data/matches'
event_path = '/Users/livio/Documents/Bachelor Thesis/python/xG Analysis/open-data/data/events/'

#create a list of every competition
def list_competitions():
    comp = []
    for root, _, files in os.walk(matches_path):
        for name in files:
            print(name)
            if name.endswith('.json'):
                comp.append(os.path.join(root, name))
    return comp

#collect every game_id and split men/women
def get_games(comp):
    men = []
    women = []
    for file_path in comp:
        print(file_path)
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                matches = json.load(f)  # Load JSON data from the file

                #look at first game for gender, store all the match ids and assign them to men or women
                gender = 0 if matches[0]['home_team']['home_team_gender'] == 'male' else 1
                match_ids = [match['match_id'] for match in matches]

                if gender:
                    women.extend(match_ids)
                else:
                    men.extend(match_ids)

    return women, men

#collect every shot and store relevant data with it
def get_shots(games):
    shot_data = []
    for game in games:
        path = event_path + str(game) + '.json'
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)  # Load JSON data from the file
            for event in data:
                #check if event is a shot and additionally get rid of penalties
                if event['type']['name'] == 'Shot' and event['shot']['type']['name'] != 'Penalty':
                    #calculate distance based on the shooting location
                    x, y = event['location']
                    dis = yard_to_meter(get_distance(x, y))

                    #calculate the angle and displacement
                    angle, displacement = get_angle(x, y)
                    vis_angle = get_visible_angle(x, y, event['shot']['freeze_frame'], angle)

                    if vis_angle > angle:
                        print(f"{vis_angle} and {angle}")

                    shot_info = {
                        'location': event['location'],
                        'distance': dis,
                        'angle': angle,
                        'end_location': event['shot']['end_location'],
                        'goal': 1 if event['shot']['outcome']['name'] == 'Goal' else 0,
                        'xG': event['shot']['statsbomb_xg'],
                        'freeze_frame': event['shot']['freeze_frame'],
                        'type': event['shot']['type']['name'],
                        'displacement': displacement,
                        'vis_angle': vis_angle
                    }
                    shot_data.append(shot_info)
                    print(shot_info)

    shot_df = pd.DataFrame(shot_data)
    return shot_df

def get_data():
    comp = list_competitions()
    women_ids, men_ids = get_games(comp)

    men_df = get_shots(men_ids)
    women_df = get_shots(women_ids)

    #basic stats
    print(f"number of competitions: {len(comp)}")
    print(f"number of games: {len(women_ids)+len(men_ids)} ({len(men_ids)}, {len(women_ids)}")
    print(f"number of shots: {len(women_df)+len(men_df)} ({len(men_df)}, {len(women_df)}")

    return men_df, women_df

def main():
    get_data()

if __name__ == "__main__":
    main()