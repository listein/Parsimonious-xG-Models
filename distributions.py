from getdata import*

plt.rcParams.update({
    'font.size': 24,           # Base font size
    'axes.titlesize': 28,      # Title font size
    'axes.labelsize': 24,      # x and y label font size
    'xtick.labelsize': 24,     # x-axis tick font size
    'ytick.labelsize': 24,     # y-axis tick font size
})

# distribution distance - shots/goals
def distance_distr(shot_df, goal, gender):
    output_folder = '/Users/livio/Documents/img_BT/test/distance'
    os.makedirs(output_folder, exist_ok=True)

    #test different bin sizes for the graph
    bins_list = [80, 120, 160, 240]
    for i, bins in enumerate(bins_list):
        #create figure with distribution
        plt.figure(figsize=(12, 8))
        plt.hist(shot_df['distance'], bins=bins, color='#00796b' if gender=='Men' else '#ff7043', edgecolor='#00796b' if gender=='Men' else '#ff7043')

        #create title and labels
        plt.xlabel('Distance [m]')
        ylabel = f'Number of {'goals scored' if goal else 'shots taken'}'
        plt.ylabel(ylabel)
        title = f'{'Men:' if gender=='Men' else 'Women:'} Distance Distribution {'(only Goals)' if goal else ''}'
        plt.title(title)

        filename = os.path.join(output_folder, f"{gender[0]}dis{'g' if goal else ''}{i}.pdf")
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

# distribution shooting angle - shots/goals
def shot_angle_distr(shot_df, goal, gender):
    output_folder = '/Users/livio/Documents/img_BT/test/shot_angle'
    os.makedirs(output_folder, exist_ok=True)

    #test different bin sizes for the graph
    bins_list = [120, 150, 180, 210, 240, 270]
    for i, bins in enumerate(bins_list):
        #create figure with distribution
        plt.figure(figsize=(12, 8))
        plt.hist(shot_df['angle'], bins=bins, range=(0, 180), color='#00796b' if gender=='Men' else '#ff7043', edgecolor='#00796b' if gender=='Men' else '#ff7043')

        #create title and labels
        plt.xlabel('Shooting angle')
        ylabel = f'Number of {'goals scored' if goal else 'shots taken'}'
        plt.ylabel(ylabel)
        title = f'{'Men:' if gender=='Men' else 'Women:'} Shooting Angle Distribution {'(only Goals)' if goal else ''}'
        plt.title(title)

        filename = os.path.join(output_folder, f"{gender[0]}ang{'g' if goal else ''}{i}.pdf")
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

# distribution scoring potential - shots/goals
def scoring_potential_distr(shot_df, goal, gender):
    # create figure with distribution
    output_folder = '/Users/livio/Documents/img_BT/test/scoring_potential'
    os.makedirs(output_folder, exist_ok=True)

    #get scoring potential for each shot: pot = angle/distance
    pot_df = shot_df['angle'] / shot_df['distance']

    #test different bin sizes for the graph
    bins_list = [40, 80, 120, 160, 180, 210, 240, 270]
    for i, bins in enumerate(bins_list):
        #create figure with distribution
        plt.figure(figsize=(12, 8))
        plt.hist(pot_df, bins=bins, range=(0, 40), color='#00796b' if gender=='Men' else '#ff7043', edgecolor='#00796b' if gender=='Men' else '#ff7043')

        #create title and labels
        plt.xlabel('Scoring potential')
        ylabel = f'Number of {'goals scored' if goal else 'shots taken'}'
        plt.ylabel(ylabel)
        title = f'{'Men:' if gender=='Men' else 'Women:'} Scoring Potential Distribution {'(only Goals)' if goal else ''}'
        plt.title(title)

        filename = os.path.join(output_folder, f"{gender[0]}pot{'g' if goal else ''}{i}.pdf")
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

# distribution scoring visible angle - shots/goals
def vis_ang_distr(shot_df, goal, gender):
    output_folder = '/Users/livio/Documents/img_BT/test/visible_angle'
    os.makedirs(output_folder, exist_ok=True)

    #test different bin sizes for the graph
    bins_list = [120, 150, 180, 210, 240, 270]
    for bins in bins_list:
        #create figure with distribution
        plt.figure(figsize=(12, 8))
        plt.hist(shot_df['vis_angle'], bins=bins, range=(0, 180), color='#00796b' if gender=='Men' else '#ff7043', edgecolor='#00796b' if gender=='Men' else '#ff7043')

        #create title and labels
        plt.xlabel('Visible angle')
        ylabel = f'Number of {'goals scored' if goal else 'shots taken'}'
        plt.ylabel(ylabel)
        title = f'{'Men:' if gender=='Men' else 'Women:'} Visible Angle Distribution {'(only Goals)' if goal else ''}'
        plt.title(title)

        filename = os.path.join(output_folder, f"{gender}_vis{'g' if goal else 's'}_{bins}.pdf")
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

# distribution of StatsBomb xG values
def xg_distr(shot_df, goal, gender):
    output_folder = '/Users/livio/Documents/img_BT/test/xG'
    os.makedirs(output_folder, exist_ok=True)

    #test different bin sizes for the graph
    bins_list = [100, 200, 250, 500, 1000]
    for i, bins in enumerate(bins_list):
        #create figure with distribution
        plt.figure(figsize=(12, 8))
        plt.hist(shot_df['xG'], bins=bins, range=(0, 1), color='#00796b' if gender=='Men' else '#ff7043', edgecolor='#00796b' if gender=='Men' else '#ff7043')

        #create title and labels
        plt.xlabel('xG prediction')
        ylabel = f'Number of {'goals scored' if goal else 'shots taken'}'
        plt.ylabel(ylabel)
        title = f'{'Men:' if gender=='Men' else 'Women:'} xG Distribution {'(only Goals)' if goal else ''}'
        plt.title(title)

        filename = os.path.join(output_folder, f"{gender[0]}xG{'g' if goal else ''}_{i}.pdf")
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

# Probability distribution for scoring based on distance
def prob_dis_distr(shot_df, gender):
    output_folder = '/Users/livio/Documents/img_BT/test/distance'
    os.makedirs(output_folder, exist_ok=True)

    #sort shots based on distance
    shot_df = shot_df.sort_values(by='distance')

    #test different bin sizes for the graph
    bins_list = [10, 20, 40, 80, 120, 160, 200, 240, 280]
    for i, bins in enumerate(bins_list):
        #group shots into different bins
        shot_df['group'] = (np.arange(len(shot_df)) // bins)

        #for each bin: get mean distance, number of goals and shots
        grouped = shot_df.groupby('group').agg(
            mean_distance=('distance', 'mean'),
            goals=('goal', 'sum'),
            total_shots=('goal', 'count')
        ).reset_index()

        #calculate probability for scoring for each bin
        grouped['probability'] = grouped['goals'] / grouped['total_shots']

        #create figure with distribution
        plt.figure(figsize=(12, 8))
        plt.scatter(grouped['mean_distance'], grouped['probability'], color='#00796b' if gender=='Men' else '#ff7043')

        #create title and labels
        plt.xlabel('Distance [m]')
        plt.ylabel('Probability of scoring')
        title = f'{'Men:' if gender=='Men' else 'Women:'} Scoring Probability by Distance'
        plt.title(title)

        plt.ylim(0, 1)

        filename = os.path.join(output_folder, f"{gender[0]}pdis{i}.pdf")
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

# Probability distribution for scoring based on shooting angle
def prob_shot_ang_distr(shot_df, gender):
    output_folder = '/Users/livio/Documents/img_BT/test/shot_angle'
    os.makedirs(output_folder, exist_ok=True)

    #sort shots based on angle
    shot_df = shot_df.sort_values(by='angle')

    #test different bin sizes for the graph
    bins_list = [10, 20, 40, 80, 120, 160, 200, 240, 280]
    for i, bins in enumerate(bins_list):
        #group shots into different bins
        shot_df['group'] = (np.arange(len(shot_df)) // bins)

        #for each bin: get mean shooting angle, number of goals and shots
        grouped = shot_df.groupby('group').agg(
            mean_angle=('angle', 'mean'),
            goals=('goal', 'sum'),
            total_shots=('goal', 'count')
        ).reset_index()

        #calculate probability for scoring for each bin
        grouped['probability'] = grouped['goals'] / grouped['total_shots']

        #create figure with distribution
        plt.figure(figsize=(12, 8))
        plt.scatter(grouped['mean_angle'], grouped['probability'], color='#00796b' if gender=='Men' else '#ff7043')

        #create title and labels
        plt.xlabel('Angle')
        plt.ylabel('Probability of scoring')
        title = f'{'Men:' if gender=='Men' else 'Women:'} Scoring Probability by Shooting Angle'
        plt.title(title)

        plt.ylim(0, 1)

        filename = os.path.join(output_folder, f"{gender[0]}pang{i}.pdf")
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

# Probability distribution for scoring based on scoring potential
def prob_pot_distr(shot_df, gender):
    output_folder = '/Users/livio/Documents/img_BT/test/scoring_potential'
    os.makedirs(output_folder, exist_ok=True)

    #calculate scoring potential and sort shots based on it
    shot_df['pot'] = shot_df['angle'] / shot_df['distance']
    shot_df = shot_df.sort_values(by='pot')

    #test different bin sizes for the graph
    bins_list = [10, 20, 40, 80, 120, 160, 200, 240, 280]
    for i, bins in enumerate(bins_list):
        #group shots into different bins
        shot_df['group'] = (np.arange(len(shot_df)) // bins)

        #for each bin: get mean scoring potential, number of goals and shots
        grouped = shot_df.groupby('group').agg(
            mean_pot=('pot', 'mean'),
            goals=('goal', 'sum'),
            total_shots=('goal', 'count')
        ).reset_index()

        #calculate probability for scoring for each bin
        grouped['probability'] = grouped['goals'] / grouped['total_shots']

        #create figure with distribution
        plt.figure(figsize=(12, 8))
        plt.scatter(grouped['mean_pot'], grouped['probability'], color='#00796b' if gender=='Men' else '#ff7043')

        #create title and labels
        plt.xlabel('Scoring potential')
        plt.ylabel('Probability of scoring')
        title = f'{'Men:' if gender == 'Men' else 'Women:'} Scoring Probability by Scoring Potential'
        plt.title(title)

        plt.ylim(0, 1)
        plt.xlim(0, 40)

        filename = os.path.join(output_folder, f"{gender[0]}ppot{i}.pdf")
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

# Probability distribution for scoring based on visible angle
def prob_vis_angle_distr(shot_df, gender):
    output_folder = '/Users/livio/Documents/img_BT/test/visible_angle'
    os.makedirs(output_folder, exist_ok=True)

    #sort shots based on distance
    shot_df = shot_df.sort_values(by='vis_angle')

    #test different bin sizes for the graph
    bins_list = [120, 150, 180, 210, 240, 270]
    for i, bins in enumerate(bins_list):
        #group shots into different bins
        shot_df['group'] = (np.arange(len(shot_df)) // bins)

        #for each bin: get mean visible angle, number of goals and shots
        grouped = shot_df.groupby('group').agg(
            mean_vis=('vis_angle', 'mean'),
            goals=('goal', 'sum'),
            total_shots=('goal', 'count')
        ).reset_index()

        #calculate probability for scoring for each bin
        grouped['probability'] = grouped['goals'] / grouped['total_shots']

        #create figure with distribution
        plt.figure(figsize=(12, 8))
        plt.scatter(grouped['mean_vis'], grouped['probability'], color='#00796b' if gender=='Men' else '#ff7043')

        #create title and labels
        plt.xlabel('Visible angle')
        plt.ylabel('Probability of scoring')
        title = f'{'Men:' if gender == 'Men' else 'Women:'} Scoring Probability by Visible Angle'
        plt.title(title)

        plt.ylim(0, 1)

        filename = os.path.join(output_folder, f"{gender[0]}pvis{i}.pdf")
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

# this code is used to create every the distributions
def main():
    m_shot, w_shot = get_data()

    #shot/goal distributions for men and women
    distance_distr(m_shot, 0, 'Men')
    distance_distr(m_shot[m_shot['goal'] == 1], 1, 'Men')
    distance_distr(w_shot, 0, 'Women')
    distance_distr(w_shot[w_shot['goal'] == 1], 1, 'Women')

    shot_angle_distr(m_shot, 0, 'Men')
    shot_angle_distr(m_shot[m_shot['goal'] == 1], 1, 'Men')
    shot_angle_distr(w_shot, 0, 'Women')
    shot_angle_distr(w_shot[w_shot['goal'] == 1], 1, 'Women')

    scoring_potential_distr(m_shot, 0, 'Men')
    scoring_potential_distr(m_shot[m_shot['goal'] == 1], 1, 'Men')
    scoring_potential_distr(w_shot, 0, 'Women')
    scoring_potential_distr(w_shot[w_shot['goal'] == 1], 1, 'Women')

    vis_ang_distr(m_shot, 0, 'Men')
    vis_ang_distr(m_shot[m_shot['goal'] == 1], 1, 'Men')
    vis_ang_distr(w_shot, 0, 'Women')
    vis_ang_distr(w_shot[w_shot['goal'] == 1], 1, 'Women')

    xg_distr(m_shot, 0, 'Men')
    xg_distr(m_shot[m_shot['goal'] == 1], 1, 'Men')
    xg_distr(w_shot, 0, 'Women')
    xg_distr(w_shot[w_shot['goal'] == 1], 1, 'Women')

    #probability distribution for men and women
    prob_dis_distr(m_shot, 'Men')
    prob_dis_distr(w_shot, 'Women')

    prob_shot_ang_distr(m_shot, 'Men')
    prob_shot_ang_distr(w_shot, 'Women')

    prob_pot_distr(m_shot, 'Men')
    prob_pot_distr(w_shot, 'Women')

    prob_vis_angle_distr(m_shot, 'Men')
    prob_vis_angle_distr(w_shot, 'Women')

if __name__ == "__main__":
    main()