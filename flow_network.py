from collections import defaultdict
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import networkx as nx
from collections import Counter
import math
import scipy.optimize
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

def load_dataset(tournament):
    """
    Unpack the dataset from Wyscout into separate dataframes. 
    
    Parameters:
    ----------
    tournament : string
        name of the soccer tournament to be analysed, as denoted in Wyscout.
    
    Returns:
    --------
    matches : pd.DataFrame() 
        Contains all the football matches of the tournament.
    events : pd.DataFrame()
        Contains all the events for all football matches.
    players : pd.DataFrame()
        Contains information on all the players enlisted for the tournament.
    competitions : pd.DataFrame()
        Contains an overview of the various competitions of the Wyscout dataset.
    teams : pd.DataFrame()
        An overview of the teams in the Wyscout dataset.
    """
    matches, events = {}, {}
    matches = pd.read_json("./data/matches/matches_{}.json".format(tournament).replace("'", '"')).iloc[::-1].reset_index()
    events = pd.read_json("./data/events/events_{}.json".format(tournament).replace("'", '"'))
    players = pd.read_json("./data/players.json".replace("'", '"'))
    teams = pd.read_json("./data/teams.json".replace("'", '"'))
    competitions = pd.read_json(open('./data/competitions.json'))
    return matches, events, players, competitions, teams

def splitter(matches, events):
    """
    Split the events DataFrame into a list containing the events sorted per match.
    
    Parameters:
    -----------
    matches : pd.DataFrame()
        Contains all the football matches of the tournament.
    events : pd.DataFrame()
        Contains all the events for all football matches.
        
    Returns:
    --------
    all_events : list
        Each element of the list corresponds with a DataFrame containing all the events of one match.
    """
    # split events into a list containing a dataframe per match
    match_id = matches["wyId"].to_numpy()
    events = events[["playerId", "matchId", "eventName", "teamId"]]
    event_id = events["matchId"].to_numpy()
    all_events = []
    for label in match_id:
        match = pd.DataFrame(columns = ["playerId", "eventName", "teamId"])
        k=0
        for i in range(len(event_id)):
            if label == event_id[i]:
                match.loc[k, :] = events.loc[i, ["playerId", "eventName", "teamId"]] 
                k = k+1
        all_events.append(match) # assume dataset is clustered per match
    return all_events

def sorter(all_events, Y):
    """
    Filters down the events of every match in the first Y matches of all_events to contain only the passes
    or shots and the event directly after a passing, respectively.
    
    Parameters:
    -----------
    all_events : list
        Each element of the list corresponds with a DataFrame containing all the events of one match.
    Y : int
        The first number of matches in all_events to filter to only contain passes and the event directly
        after a passing.
        
    Returns:
    --------
    meta_passes : list
        Each entry corresponds to a DataFrame containing the passes and the event directly after passing
        for one match.
    meta_shots: list
        Each entry corresponds to a DataFrame containing the shots for one match.   
    """
    first_events = all_events[:Y]
    meta_passes = []
    meta_shots = []
    
    # select the passes + event after: obtain succesful (y/n) + path
    for i in range(len(first_events)):
        pass_index = first_events[i].index[first_events[i]['eventName'] == "Pass"].to_numpy()
        next_index = pass_index+1
        tot_index = np.unique(np.concatenate((pass_index, next_index), axis = 0))
        # prevent indexing after last pass if its the last element
        # this happens when 
        tot_index = tot_index[tot_index <= len(first_events[i])-1]
        shot_index = first_events[i].index[first_events[i]['eventName'] == "Shot"].to_numpy()
        shots = pd.DataFrame(columns = ["playerId", "eventName", "teamId"])
        passes = pd.DataFrame(columns = ["playerId", "eventName", "teamId"])
        for c, el in enumerate(tot_index):
            passes.loc[c,:] = first_events[i].loc[el, ["playerId", "eventName", "teamId"]]
        for c, el in enumerate(shot_index):
            shots.loc[c,:] = first_events[i].loc[el, ["playerId", "eventName", "teamId"]]
        meta_passes.append(passes)
        meta_shots.append(shots)
    return meta_passes, meta_shots

def check_pass(df):
    """
    Check if the pass is a succesful pass or a failed pass. 
    
    Parameters:
    -----------
    df : pd.DataFrame()
        Contains the passes and event directly after passing for one match.
        
    Returns:
    --------
    succesful : pd.DataFrame()
        Contains the succesful passes of one match.
    fails : pd.DataFrame()
        Contains the failed passes of one match.
    """
    succesful = pd.DataFrame(columns = ["From", "To", "teamId"])
    fails = pd.DataFrame(columns = ["Player"])
    for c, i in enumerate(range(1, len(df))):
        if df["teamId"][i-1] == df["teamId"][i]:
            succesful.loc[c,:] = [df["playerId"][i-1], df["playerId"][i], df["teamId"][i]]
        else:
            fails.loc[c,:] = df["playerId"][i-1]
    succesful.reset_index(drop=True, inplace=True)
    return succesful, fails

def sort_passes(df, matches, Z):
    """
    Sort passes into the passes of teamA and teamB for match Z.
    
    Parameters:
    -----------
    df : pd.DataFrame()
        Contains the passes of one match.
    matches : pd.DataFrame() 
        Contains all the football matches of the tournament.
    Z : int
        Match number of the tournament.
        
    Returns:
    --------
    teamA : pd.DataFrame()
        Passes of teamA.
    teamB : pd.DataFrame()
        Passes of teamB.
    players_teamA: pd.DataFrame()
        players of teamA.
    players_teamB: pd.DataFrame()
        players of teamB.
    """
    teams = list(matches["teamsData"][Z].keys())
    
    players_teamA = players.loc[players['currentNationalTeamId'] == int(teams[0])]
    players_teamB =  players.loc[players['currentNationalTeamId'] == int(teams[1])]
    tms = df.groupby("teamId").groups
    teamA = pd.DataFrame(columns = ["From", "To"])
    teamB = pd.DataFrame(columns = ["From", "To"])
    dfs = [teamA, teamB]

    # separate succesful df into two teams
    for i in range(len(dfs)):
        df_team = dfs[i]
        team = teams[i]
        for x,y in tms.items():
            if int(x) == int(team):
                for c, j in enumerate(y):
                    To = succesful.iloc[j,0]
                    From = succesful.iloc[j,1]
                    row = [To, From]
                    df_team.loc[c] = row
    return teamA, teamB, players_teamA, players_teamB

def analyse_passes(succesful):
    """
    Analyse the succesful passes of one team and get some statistics. These include: the number of 
    passes per player, the tuples of players involved in one pass and how often player A 
    and B pass to each other.
    
    Parameters:
    -----------
    succesful : pd.DataFrame()
    
    Returns:
    --------
    passes_per_player : dictionary
        The keys correspond with the player_id and the values are lists with indices in the passes 
        DataFrame where the player has scored.
    permutations : list
        Each entry corresponds with a tuple of two player_ids. The first element of a tuple is the 
        player initiating a pass, the second element of the tuple is the player receiving the pass.
    count : Counter() (dictionary)
        Each key is a tuple from the permutations list and each value is the number of times this
        tuple occurs (how often two players pass to each other). Note (a,b) != (b,a) here.    
    links : dictionary
        The keys correspond with the tuples from permutations and the values correspond with lists
        of indices in the passing DataFrame where this pass has occured.
    """
    links = succesful.groupby(["From", "To"]).groups
    passes_per_player = succesful.groupby(["From"]).groups
    permutations = list(links.keys())
    count = Counter(tuple(sorted(t)) for t in permutations)
    return passes_per_player, permutations, count, links

def get_arcs(permutations, links):
    """
    Calculate the weights of the arcs of the flow diagram. These correspond with the number of passes 
    between each player for one team during one match.
    
    Parameters:
    -----------
    permutations : list
        Each entry corresponds with a tuple of two player_ids. The first element of a tuple is the 
        player initiating a pass, the second element of the tuple is the player receiving the pass.
    links : dictionary
        The keys correspond with the tuples from permutations and the values correspond with lists
        of indices in the passing DataFrame where this pass has occured.
        
    Returns:
    --------
    arcs : pd.DataFrame()
        Contains the pairs of players involved in passes and the number of passes for such a pair.
        Note that (a,b) = (b,a) in this case.
    """
    doubles = pd.DataFrame(columns = ['tuples1', 'tuples2', 'index1', 'index2'])
    for c, i in enumerate(range(len(permutations))):
        a = permutations[i][0]
        b = permutations[i][1]
        if (b,a) in permutations:
            index = permutations.index((b,a))
            row = [(a,b), (b,a), c, index]
            doubles.loc[c,:] = row
            
    doubles = doubles.query('index1!= index2')        
    # get weights of the arcs
    arcs = pd.DataFrame(columns = ['tuples','arc weight'])
    for c, i in enumerate(range(len(doubles['index1']))):
        tuples1 = doubles.iloc[i,0]
        tuples2 = doubles.iloc[i,1]
        passes1 = len(links[tuples1])
        passes2 = len(links[tuples2])
        passes = passes1+passes2
        row = [tuples1, passes]
        arcs.loc[c,:] = row
    arcs = arcs[arcs['arc weight']>5]
    return arcs


def sort_positions(arcs, players_df):
    """
    This function sorts the players of a team into goalkeeper, defender, midfielder or attacker.
    
    Parameters:
    -----------
    arcs : pd.DataFrame()
        Contains the pairs of players involved in passes and the number of passes for such a pair.
        Note that (a,b) = (b,a) in this case.
    players_df : pd.DataFrame()
        Contains the players of one team during one match.
        
    Returns:
    --------
    positions : dictionary
        The keys correspond with the function of the player and the values are lists corresponding 
        with the player_ids fulfilling said function.
    """
    positions = {'Goalkeeper':[], 'Defender':[], 'Midfielder':[], 'Forward':[]}
    arr = np.zeros(len(arcs['tuples'])*2)
    for i in range(len(arcs['tuples'])):
        arr[i], arr[i+1] = list(arcs['tuples'])[i][0], list(arcs['tuples'])[i][1]
    arr = np.unique(arr[arr != 0]) # all team players
    for j in arr:
        index = np.where(players_df['wyId'] == j)[0][0]
        name = players_df.iloc[index,8]['name']
        positions[name].append(int(j))
    return positions

def draw(positions, arcs, matches, teams, A):
    """
    Draw graph with team strengths as arcs.
    
    Parameters:
    -----------
    positions : dictionary
        The keys correspond with the function of the player and the values are lists corresponding 
        with the player_ids fulfilling said function.
    arcs : pd.DataFrame()
        Contains the pairs of players involved in passes and the number of passes for such a pair.
        Note that (a,b) = (b,a) in this case.
    matches : pd.DataFrame() 
        Contains all the football matches of the tournament.
    teams : pd.DataFrame()
        An overview of the teams in the Wyscout dataset.
    A : int
        Team A or Team B.
    Returns:
    --------
    None
    """
    subset_sizes = [len(i) for i in positions.values()]
    
    # map out positions
    pos = {n: (0, 1/len(positions['Defender'])) for i,n in enumerate(positions['Goalkeeper'])}
    pos.update({n: (1, (i)/len(positions['Defender'])) for i, n in enumerate(positions['Defender'])})
    if len(positions['Defender']) == 4:
        pos[positions['Defender'][0]]= (1.2, 0)
        pos[positions['Defender'][1]]= (1, 0.25)
        pos[positions['Defender'][2]]= (1, 0.5)
        pos[positions['Defender'][3]]= (1.2, 0.75)
        
    if len(positions['Defender']) == 5:
        pos[positions['Defender'][0]]= (1.3, 0)
        pos[positions['Defender'][1]]= (1.2, 0.15)
        pos[positions['Defender'][2]]= (1, 0.3)
        pos[positions['Defender'][3]]= (1.2, 0.45)    
        pos[positions['Defender'][4]]= (1.3, 0.6)   
    pos.update({n: (2, (i)/len(positions['Midfielder'])) for i, n in enumerate(positions['Midfielder'])})
    if len(positions['Midfielder']) == 4:
        pos[positions['Midfielder'][0]]= (2.2, 0)
        pos[positions['Midfielder'][1]]= (2, 0.25)
        pos[positions['Midfielder'][2]]= (2, 0.5)
        pos[positions['Midfielder'][3]]= (2.2, 0.75)    
    pos.update({n: (3, i/len(positions['Forward'])) for i, n in enumerate(positions['Forward'])})
    weighted_edges = dict(zip(list(arcs['tuples']), list(arcs['arc weight'])))
    weighted_edges
    G = nx.Graph()
    G.add_edges_from(weighted_edges)
    for e in G.edges():
        G[e[0]][e[1]]['weight'] = weighted_edges[e]
        
    widths = nx.get_edge_attributes(G, 'weight')
    labels = {e: G.edges[e]['weight'] for e in G.edges}
    nodelist = G.nodes()
    
    plt.figure(figsize=(18,12))
    plt.xlim(-0.1,3.1)
    plt.ylim(-0.1,1)
    #plt.imshow(img, aspect='auto')
    nx.draw_networkx_nodes(G,pos,
                           nodelist=nodelist,
                           node_size=1500,
                           node_color='black',
                           alpha=0.7)
    nx.draw_networkx_edges(G,pos,
                           edgelist = widths.keys(),
                           width=list(widths.values()),
                           edge_color='lightblue',
                           alpha=0.6)
    nx.draw_networkx_labels(G, pos=pos,
                            labels=dict(zip(nodelist,nodelist)),
                            font_color='white')
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    
    nx.spring_layout(G, k=1, iterations=20)
    plt.box(False)
    team_id = int(list(matches["teamsData"][cherry].keys())[A])
    team_name = teams.iloc[np.where(teams['wyId']==team_id)[0][0],1]
    plt.title("Flow Network " + str(team_name))