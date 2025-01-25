from getdata import*
from mplsoccer import Pitch, VerticalPitch
from scipy.ndimage import gaussian_filter
import os
from auxiliaries import*

plt.rcParams.update({
    'font.size': 24,           # Base font size
    'axes.titlesize': 28,      # Title font size
    'axes.labelsize': 24,      # x and y label font size
    'xtick.labelsize': 24,     # x-axis tick font size
    'ytick.labelsize': 24,     # y-axis tick font size
})

#create a heatmap by counting shots and group them into bins
def heatmap(df_shots, gender):
    output_folder = '/Users/livio/Documents/img_BT/test/heatmaps'
    os.makedirs(output_folder, exist_ok=True)

    #filter goals and save every coordinate of shots/goals
    df_goals = df_shots[df_shots['goal']==1]
    x_s, y_s = zip(*df_shots['location'])
    x_g, y_g = zip(*df_goals['location'])

    # create different bins
    bins = [(480, 320), (360, 240), (240, 160), (120, 80), (60, 40), (45, 30), (30, 20)]
    for ctr, (i, j) in enumerate(bins):
        #create pitch
        pitch = VerticalPitch(pitch_type='statsbomb', half=True,
                          pitch_color='white', line_color='black', line_zorder=2,
                          axis=True, label=True, tick=True)

        fig, ax = pitch.draw(figsize=(12, 8))

        ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False)
        ax.tick_params(axis='y', left=True, right=False, labelleft=True, labelright=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        #create bins for shot data
        bin_stat_shot = pitch.bin_statistic(x_s, y_s, statistic='count', bins=(i, j))
        bin_stat_shot['statistic'] = gaussian_filter(bin_stat_shot['statistic'], 1)
        pcm = pitch.heatmap(bin_stat_shot, ax=ax, cmap='bone_r' if gender=='m' else 'gist_heat_r' , edgecolors=None) #hot_r

        plt.title(f'{'Men:' if gender=='m' else 'Women:'} Heatmap for Every Shot')
        filename = os.path.join(output_folder, f"{gender}{ctr}h.pdf")
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        print(f"Saved: {filename}")

        #create bins for goal data
        bin_stat_goal = pitch.bin_statistic(x_g, y_g, statistic='count', bins=(i, j))
        bin_stat_goal['statistic'] = gaussian_filter(bin_stat_goal['statistic'], 1)
        pcm = pitch.heatmap(bin_stat_goal, ax=ax, cmap='bone_r' if gender=='m' else 'gist_heat_r', edgecolors=None)  # hot_r

        plt.title(f'{'Men:' if gender=='m' else 'Women:'} Heatmap for Every Goal scored')
        filename = os.path.join(output_folder, f"{gender}{ctr}hg.pdf")
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

def angle_presentation():
    output_folder = '/Users/livio/Documents/img_BT/test/shot_angle'
    os.makedirs(output_folder, exist_ok=True)

    #create coordinates for whole pitch
    x_coord = np.linspace(0, 120, 300)
    y_coord = np.linspace(0, 80, 200)

    # Initialize the angle grid
    angles = np.zeros((len(y_coord), len(x_coord)))

    # Calculate the angle for each grid point
    for i, x in enumerate(x_coord):
        for j, y in enumerate(y_coord):
            angles[j, i], _ = get_angle(x, y)

    #levels which will be plotted
    levels = np.concatenate([np.arange(5, 35, 5), [40, 60, 90, 150]])

    #create pitch
    pitch = Pitch(pitch_type='statsbomb', half=True,
                          pitch_color='white', line_color='black', line_zorder=2,
                          axis=True, label=True, tick=True)

    fig, ax = pitch.draw(figsize=(12, 8))

    ax.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True, labelrotation=-90)
    ax.tick_params(axis='y', left=True, right=False, labelleft=True, labelright=False, labelrotation=-90)

    ax.set_xticks(range(60, 121, 20))
    ax.set_yticks(range(0, 81, 20))

    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Plot the contour lines
    contour = ax.contour(x_coord, y_coord, angles, levels=levels, colors='black', linewidths=1)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%1.0f°')

    filename = os.path.join(output_folder,  "anglev.pdf")
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def distance_presentation():
    output_folder = '/Users/livio/Documents/img_BT/test/distance'
    os.makedirs(output_folder, exist_ok=True)

    x_coord = np.linspace(0, 120, 300)
    y_coord = np.linspace(0, 80, 200)

    # Initialize the angle grid
    distances = np.zeros((len(y_coord), len(x_coord)))

    # Calculate the angle for each grid point
    for i, x in enumerate(x_coord):
        for j, y in enumerate(y_coord):
            distances[j, i] = yard_to_meter(get_distance(x, y))

    #levels which will be plotted
    levels = np.concatenate([np.arange(0, 60, 5)])

    pitch = Pitch(pitch_type='statsbomb', half=True,
                  pitch_color='white', line_color='black', line_zorder=2,
                  axis=True, label=True, tick=True)

    fig, ax = pitch.draw(figsize=(12, 8))

    ax.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True, labelrotation=-90)
    ax.tick_params(axis='y', left=True, right=False, labelleft=True, labelright=False, labelrotation=-90)

    ax.set_xticks(range(60, 121, 20))
    ax.set_yticks(range(0, 81, 20))

    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Plot the contour lines
    contour = ax.contour(x_coord, y_coord, distances, levels=levels, colors='black', linewidths=1)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%1.0fm')

    filename = os.path.join(output_folder, "dislev.pdf")
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def potential_presentation():
    output_folder = '/Users/livio/Documents/img_BT/test/scoring_potential'
    os.makedirs(output_folder, exist_ok=True)

    x_coord = np.linspace(0, 120, 300)
    y_coord = np.linspace(0, 80, 200)

     # Initialize the angle grid
    potential = np.zeros((len(y_coord), len(x_coord)))


     # Calculate the angle for each grid point
    for i, x in enumerate(x_coord):
        for j, y in enumerate(y_coord):
            angle, _ = get_angle(x, y)
            distance = yard_to_meter(get_distance(x, y))
            potential[j, i] = angle/distance

        # levels which will be plotted
    #levels = np.concatenate([np.arange(0.1, 0.55, 0.1), [1, 2, 5, 10, 50]])
    levels = np.concatenate([[2, 3, 4, 5, 7.5, 10, 15, 50]])

    pitch = Pitch(pitch_type='statsbomb', half=True,
                  pitch_color='white', line_color='black', line_zorder=2,
                  axis=True, label=True, tick=True, line_alpha=0.3, goal_alpha=0.3, linestyle='--')

    fig, ax = pitch.draw(figsize=(12, 8))

    ax.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True, labelrotation=-90)
    ax.tick_params(axis='y', left=True, right=False, labelleft=True, labelright=False, labelrotation=-90)

    ax.set_xticks(range(60, 121, 20))
    ax.set_yticks(range(0, 81, 20))

    ax.set_xlim(97, 120)
    ax.set_ylim(25, 55)

    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Plot the contour lines
    contour = ax.contour(x_coord, y_coord, potential, levels=levels, colors='black', linewidths=1)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')

    filename = os.path.join(output_folder, "potlev2.pdf")
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def example_distance():
    output_folder = '/Users/livio/Documents/img_BT/test/distance'
    os.makedirs(output_folder, exist_ok=True)

    pitch = Pitch(pitch_type='statsbomb', half=True,
                  pitch_color='white', line_color='black', line_zorder=2,
                  axis=True, label=True, tick=True)

    fig, ax = pitch.draw(figsize=(12, 8))

    ax.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True, labelrotation=-90)
    ax.tick_params(axis='y', left=True, right=False, labelleft=True, labelright=False, labelrotation=-90)

    ax.set_xticks(range(60, 121, 20))
    ax.set_yticks(range(0, 81, 20))

    ax.scatter(107, 32, color='black')
    ax.plot([107, 120], [32, 40], color='black')

    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)

    midpoint = (111, 35)
    ax.text(midpoint[0], midpoint[1], f'{14} m', color='black', fontsize=12, ha='center', rotation=-90)

    filename = os.path.join(output_folder, "disex2.pdf")
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def example_angle():
    output_folder = '/Users/livio/Documents/img_BT/test/shot_angle'
    os.makedirs(output_folder, exist_ok=True)

    pitch = Pitch(pitch_type='statsbomb', half=True,
                  pitch_color='white', line_color='black', line_zorder=2,
                  axis=True, label=True, tick=True)

    fig, ax = pitch.draw(figsize=(12, 8))

    ax.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True, labelrotation=-90)
    ax.tick_params(axis='y', left=True, right=False, labelleft=True, labelright=False, labelrotation=-90)

    ax.set_xticks(range(60, 121, 20))
    ax.set_yticks(range(0, 81, 20))

    ax.scatter(107, 32, color='black')
    ax.plot([107, 120], [32, 36], color='black', linestyle='--')
    ax.plot([107, 120], [32, 44], color='black', linestyle='--')

    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)

    angle, _ = get_angle(107, 32)

    ax.text(107, 30, f'{angle:.0f}°', color='black', fontsize=12, ha='center', rotation=-90)

    filename = os.path.join(output_folder, "angex.pdf")
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def main():
    men_df, women_df = get_data()

    heatmap(men_df, 'm')
    heatmap(women_df, 'w')

    example_angle()
    example_distance()
    potential_presentation()
    distance_presentation()
    angle_presentation()

if __name__ == "__main__":
    main()