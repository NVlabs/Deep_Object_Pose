import matplotlib
import pickle 
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import os 

import glob

# load the data 
# might be multiple datasets

# make the plots 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', 
    default = None, 
    help='path to data output')
parser.add_argument('--data',
    nargs='+',
    default=None,
    help='list of csv files')
parser.add_argument('--labels',
    nargs='+',
    default=None,
    help='labels to put')
parser.add_argument('--colours',
                    nargs='+',
                    default=None, 
                    help = '')

parser.add_argument("--outf",
    default=None,
    help="where to put the data")

parser.add_argument('--threshold',
                    default = 0.1,
                    type = float
                    )
parser.add_argument('--title',
                    default = 'AUC')
parser.add_argument('--filename',
                    default = 'output')
parser.add_argument('--styles',
                    nargs='+',
                    default=None, 
                    help = '')
parser.add_argument("--show",
    action='store_true',
    help="show the graph at the end. "
    )
parser.add_argument("--pixels",
    action='store_true',
    help="Using keypoint distance as metric"
    )
opt = parser.parse_args()
sns.set_style("white")
sns.set_style("ticks")
sns.set_context("paper")
# sns.set_context("notebook")
# sns.set_context("talk")
sns.despine()
# load the data 

# if folder load all the files and create a graph
# if a list put all of them in the same graph

plt.tight_layout()
# sns.set(font_scale=1.1)

if opt.data_folder is not None:
    # load the data from the file
    adds_to_load = glob.glob(f"{opt.data_folder}/adds*")        
    counts_dict = pickle.load(open(f"{opt.data_folder}/count_all_annotations.p",'rb'))

else:
    # load the files in the list 
    adds_to_load = opt.data
    counts_dict = None

    fig = plt.figure()
    ax = plt.axes()


for i_file, file in enumerate(adds_to_load):
    print(file)
    label = file.split("/")[-1]
    label = label.replace('adds_','').replace(".p",'')
    filename = label

    if not counts_dict is None: 
        fig = plt.figure()
        ax = plt.axes()
    
        n_pnp_possible_frames = counts_dict[filename]

    else:
        # check labels
        try:
            label = opt.labels[i_file]
        except:
            label = filename

        # get n possible solutions
        path = "/".join(file.split("/")[0:-1]) + '/'
        n_pnp_possible_frames = pickle.load(open(f"{path}/count_all_annotations.p",'rb'))[filename]        

    adds_objects = pickle.load(open(file,'rb'))

    # add_pnp_found = np.array(adds_objects)/100
    add_pnp_found = np.array(adds_objects)
    print('mean',add_pnp_found.mean(),'std',add_pnp_found.std(),
        'ratio',f'{len(add_pnp_found)}/{n_pnp_possible_frames}')
    n_pnp_found = len(add_pnp_found)

    delta_threshold = opt.threshold/300
    add_threshold_values = np.arange(0., opt.threshold, delta_threshold)

    counts = []
    for value in add_threshold_values:
        under_threshold = len(np.where(add_pnp_found <= value)[0])/n_pnp_possible_frames
        counts.append(under_threshold)

    for value in [0.02,0.04,0.06]:
        under_threshold = len(np.where(add_pnp_found <= value)[0])/n_pnp_possible_frames
        print('auc at ',value,':', under_threshold)
    auc = np.trapz(counts, dx = delta_threshold)/opt.threshold

    # divide might screw this up .... to check!
    print('auc',auc)
    # print('found', n_pnp_found/n_pnp_possible_frames)
    # print('mean', np.mean(add[np.where(add > pnp_sol_found_magic_number)]))
    # print('median',np.median(add[np.where(add > pnp_sol_found_magic_number)]))
    # print('std',np.std(add[np.where(add > pnp_sol_found_magic_number)]))

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if counts_dict is None:
        colour = cycle[int(opt.colours[i_file])]
        # colour = cycle[int(i_file)]
    else:
        colour = cycle[0]

    try:
        style = args.styles[i_csv]
        if style == '0':
            style = '-'
        elif style == '1':
            style = '--'
        elif style == '2':
            style = ':'

        else:
            style = '-'
    except:
        style = '-'

    label = f'{label} ({auc:.3f})'
    ax.plot(add_threshold_values, counts,style,color=colour,label=label)

    if not counts_dict is None:
        if opt.pixels:
            plt.xlabel('L2 threshold distance (pixels)')
        else:
            plt.xlabel('ADD threshold distance (m)')
        plt.ylabel('Accuracy')
        plt.title(f'{filename} auc: {auc:.3f}')

        ax.set_ylim(0,1)
        ax.set_xlim(0, float(opt.threshold))

        # ax.set_xticklabels([0,20,40,60,80,100])
        plt.tight_layout()
        plt.savefig(f'{opt.data_folder}/{filename}.png')
        plt.close()

if counts_dict is None: 
    if opt.pixels:
        plt.xlabel('L2 threshold distance (pixels)')
    else:
        plt.xlabel('ADD threshold distance (m)')

    plt.ylabel('Accuracy')
    plt.title(opt.title)
    ax.legend(loc='lower right',frameon = True, fancybox=True, framealpha=0.8)


    legend = ax.get_legend()
    for i, t in enumerate(legend.get_texts()):
        if opt.data[i] == '666':
            t.set_ha('left') # ha is alias for horizontalalignment
            t.set_position((-30,0))

    ax.set_ylim(0,1)
    ax.set_xlim(0, float(opt.threshold))
    # ax.set_xticklabels([0,20,40,60,80,100])
    plt.tight_layout()
    try:
        os.mkdir(opt.outf)
    except:
        pass
    if opt.outf is None:
        plt.savefig(f'{opt.filename}.png')
    else:
        plt.savefig(f'{opt.outf}/{opt.filename}.png')
    if opt.show:
        plt.show()
    plt.close()


