# This file runs tile-level inference on the training, calibration and internal validation cohort
import argparse
import os
import time
import h5py
import numpy as np
import pandas as pd
import geojson
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from monai.data import DataLoader, CSVDataset, PatchWSIDataset
import monai.transforms as mt

from dataset_processing import class_parser
from models import get_network

import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on slides.')

    parser.add_argument("--save_dir", default='results', help="path to folder where inference will be stored")

    #dataset processing
    parser.add_argument("--stain", choices=['he', 'qc', 'p53', 'tff3'], help="Stain to consider H&E (he), quality control (qc) or P53 (p53), TFF3(tff3)")
    parser.add_argument("--process_list", default=None, help="file containing slide-level ground truth to use.")
    parser.add_argument("--slide_path", default='slides', help="slides root folder")
    parser.add_argument("--format", default=".ndpi", help="extension of whole slide image")
    parser.add_argument("--reader", default='openslide', help="monai slide backend reader ('openslide' or 'cuCIM')")

    #model path and parameters
    parser.add_argument("--network", default='vgg_16', help="DL architecture to use")
    parser.add_argument("--model_path", required=True, help="path to stored model weights")

    parser.add_argument("--patch_path", default='patches', help="path to stored (.h5 or .csv) patch files")
    parser.add_argument("--input_size", default=None, type=int, help="size of tiles to extract")

    #data processing
    parser.add_argument("--batch_size", default=None, help="Batch size. Default is to use values set for architecture to run on 1 GPU.", type=int)
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers to call for DataLoader")
    
    parser.add_argument("--he_threshold", default=0.99993, type=float, help="A threshold for detecting gastric cardia in H&E")
    parser.add_argument("--p53_threshold", default= 0.99, type=float, help="A threshold for Goblet cell detection in p53")
    parser.add_argument("--qc_threshold", default=0.99, help="A threshold for detecting gastric cardia in H&E")
    parser.add_argument("--tff3_threshold", default= 0.93, help="A threshold for Goblet cell detection in tff3")
    parser.add_argument("--lcp_cutoff", default=None, help='number of tiles to be considered low confidence positive')
    parser.add_argument("--hcp_cutoff", default=None, help='number of tiles to be considered high confidence positive')
    parser.add_argument("--impute", action='store_true', help="Assume missing data as negative")

    #class variables
    parser.add_argument("--ranked_class", default=None, help='particular class to rank tiles by')
    parser.add_argument("--dysplasia_separate", action='store_false', help="Flag whether to separate the atypia of uncertain significance and dysplasia classes")
    parser.add_argument("--respiratory_separate", action='store_false', help="Flag whether to separate the respiratory mucosa cilia and respiratory mucosa classes")
    parser.add_argument("--gastric_separate", action='store_false', help="Flag whether to separate the tickled up columnar and gastric cardia classes")
    parser.add_argument("--atypia_separate", action='store_false', help="Flag whether to perform the following class split: atypia of uncertain significance+dysplasia, respiratory mucosa cilia+respiratory mucosa, tickled up columnar+gastric cardia classes, artifact+other")
    parser.add_argument("--p53_separate", action='store_false', help="Flag whether to perform the following class split: aberrant_positive_columnar, artifact+nonspecific_background+oral_bacteria, ignore equivocal_columnar")
    
    #outputs
    parser.add_argument("--xml", action='store_true', help='produce annotation files for ASAP in .xml format')
    parser.add_argument("--json", action='store_true', help='produce annotation files for QuPath in .geoJSON format')

    parser.add_argument('--silent', action='store_true', help='Flag which silences terminal outputs')

    args = parser.parse_args()
    return args

def torchmodify(name):
    a = name.split('.')
    for i,s in enumerate(a) :
        if s.isnumeric() :
            a[i]="_modules['"+s+"']"
    return '.'.join(a)

if __name__ == '__main__':
    args = parse_args()
    
    network = args.network
    reader = args.reader

    crf_dir = os.path.join(args.save_dir, 'crfs')
    inference_dir = os.path.join(args.save_dir, 'inference')
    annotation_dir = os.path.join(args.save_dir, 'tile_annotations')

    directories = {'slide_dir': args.slide_path, 
                   'save_dir': args.save_dir,
                   'patch_dir': args.patch_path, 
                   'inference_dir': inference_dir,
                   'annotation_dir': annotation_dir
                   }
    
    for path in directories.values():
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    print("Outputting inference to: ", directories['save_dir'])

    if args.stain == 'he':
        file_name = 'H&E'
        gt_label = 'Atypia'
        classes = class_parser('he', args.dysplasia_separate, args.respiratory_separate, args.gastric_separate, args.atypia_separate, args.p53_separate)
        if args.ranked_class is not None:
            ranked_class = args.ranked_class
        else:
            ranked_class = 'atypia'
        thresh = args.he_threshold
        mapping = {'Y': 1, 'N': 0}
        if args.lcp_cutoff is not None:
            lcp_threshold = args.lcp_cutoff
        else:
            lcp_threshold = 0 
        if args.hcp_cutoff is not None:
            hcp_threshold = args.hcp_cutoff
        else:
            hcp_threshold = 10
        channel_means = [0.7747305964175918, 0.7421753839460998, 0.7307385516144509]
        channel_stds = [0.2105364799974944, 0.2123423033814637, 0.20617556948731974]
    elif args.stain == 'qc':
        file_name = 'H&E'
        gt_label = 'QC Report'
        classes = ['Background', 'Gastric-type columnar epithelium', 'Intestinal Metaplasia', 'Respiratory-type columnar epithelium']
        mapping = {'Adequate for pathological review': 1, 'Scant columnar cells': 1, 'Squamous cells only': 0, 'Insufficient cellular material': 0, 'Food material': 0}
        if args.ranked_class is not None:
            ranked_class = args.ranked_class
        else:
            ranked_class = 'Gastric-type columnar epithelium'
        thresh = args.he_threshold
        if args.lcp_cutoff is not None:
            lcp_threshold = args.lcp_cutoff
        else:
            lcp_threshold = 0
        if args.hcp_cutoff is not None:
            hcp_threshold = args.hcp_cutoff
        else:
            hcp_threshold = 95
        channel_means = [0.485, 0.456, 0.406]
        channel_stds = [0.229, 0.224, 0.225]
    elif args.stain == 'p53':
        file_name = 'P53'
        gt_label = 'P53 positive'
        classes = class_parser('p53', args.p53_separate)
        if args.ranked_class is not None:
            ranked_class = args.ranked_class
        else:
            ranked_class = 'aberrant_positive_columnar'
        thresh = args.p53_threshold
        mapping = {'Y': 1, 'N': 0}
        if args.lcp_cutoff is not None:
            lcp_threshold = args.lcp_cutoff
        else:
            lcp_threshold = 0
        if args.hcp_cutoff is not None:
            hcp_threshold = args.hcp_cutoff
        else:
            hcp_threshold = 2
        channel_means = [0.7747305964175918, 0.7421753839460998, 0.7307385516144509]
        channel_stds = [0.2105364799974944, 0.2123423033814637, 0.20617556948731974]
    elif args.stain == 'tff3':
        file_name = 'TFF3'
        gt_label = 'TFF3 positive'
        classes = ['Equivocal', 'Negative', 'Positive']
        ranked_class = 'Positive'
        secondary_class = 'Equivocal'
        thresh = args.tff3_threshold
        mapping = {'Y': 1, 'N': 0}
        if args.lcp_cutoff is not None:
            lcp_threshold = args.lcp_cutoff
        else:
            lcp_threshold = 3
        if args.hcp_cutoff is not None:
            hcp_threshold = args.hcp_cutoff
        else:
            hcp_threshold = 40
        channel_means = [0.485, 0.456, 0.406]
        channel_stds = [0.229, 0.224, 0.225]
    else:
        raise AssertionError('args.stain must be he/qc, tff3, or p53.')

    print('Channel Means: ', channel_means, '\nChannel Stds: ', channel_stds)

    trained_model, model_params = get_network(network, class_names=classes, pretrained=False)
    try:
        trained_model.load_state_dict(torch.load(args.model_path).module.state_dict())
    except: 
        trained_model = torch.load(args.model_path)
    
    # Modify the model to use the updated GELU activation function in later PyTorch versions 
    for name, module in trained_model.named_modules():
        if isinstance(module, nn.GELU):
            exec('trained_model.'+torchmodify(name)+'=nn.GELU()')

    # Use manual batch size if one has been specified
    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        batch_size = model_params['batch_size']
    
    if args.input_size is not None:
        input_size = args.patch_size
    else:
        input_size = model_params['patch_size']
    
    if torch.cuda.is_available() and (torch.version.hip or torch.version.cuda):
        if torch.cuda.device_count() > 1:
            trained_model = torch.nn.DataParallel(trained_model, device_ids=list(range(torch.cuda.device_count())))
            device = torch.device("cuda")
        else:
            device = torch.device("cuda:0")
            trained_model.to(device)
    else:
        device = torch.device("cpu")
    trained_model.eval()

    eval_transforms = mt.Compose(
            [
                mt.ScaleIntensityRanged(keys="image", a_min=0, a_max=255, b_min=0.0, b_max=1.0),
                mt.ToTensord(keys=("image")),
                mt.TorchVisiond(keys=("image"), name="Normalize", mean=channel_means, std=channel_stds),
                mt.TorchVisiond(keys=("image"), name="Resize", size=(input_size, input_size)),
                mt.ToMetaTensord(keys=("image")),
            ]
        )

    if args.process_list is not None:
        if os.path.isfile(args.process_list):
            #TODO read labels from csv to build process list
            process_list = pd.read_csv(args.process_list, index_col=0)
            process_list.dropna(subset=[file_name], inplace=True)
            if args.impute:
                process_list[gt_label] = process_list[gt_label].fillna('N')
            else:
                process_list.dropna(subset=[gt_label], inplace=True)
            process_list.sort_index(inplace=True)
            process_list[gt_label] = process_list[gt_label].map(mapping)
        else:
            raise AssertionError('Not a valid path for ground truth labels.')
    else:
        process_list = []
        for file in os.listdir(directories['slide_dir']):
            if file.endswith(('.ndpi','.svs')):
                process_list.append(file)
        slides = sorted(process_list)
        process_list = pd.DataFrame(slides, columns=['slide_id'])
        sample_id = process_list['slide_id'].str.split(' ')
        process_list['CYT ID'] = sample_id.str[0]
        process_list['Pot ID'] = sample_id.str[2]

    records = []

    for index, row in process_list.iterrows():
        slide = row['slide_id']
        slide_name = slide.replace(args.format, "")
        try:
            wsi = os.path.join(directories['slide_dir'], slide)
        except:
            print(f'File {slide} not found.')
            continue

        if not os.path.exists(wsi):
            print(f'File {wsi} not found.')
            continue
        print(f'\rProcessing case {index+1}/{len(process_list)} {row["CYT ID"]}: ', end='')

        slide_output = os.path.join(directories['inference_dir'], slide_name)
        if os.path.isfile(slide_output + '.csv'):
            print(f'Inference for {row["CYT ID"]} already exists.')
            predictions = pd.read_csv(slide_output+'.csv')
        else:
            patch_path = os.path.join(directories['patch_dir'], slide_name+'.h5')
            patch_file = h5py.File(patch_path)['coords']
            patch_size = patch_file.attrs['patch_size']
            patch_level = patch_file.attrs['patch_level']

            locations = pd.DataFrame(np.array(patch_file), columns=['x_min','y_min'])
            locations['image'] = wsi
            print('Number of tiles:',len(locations))
            #monai coordinates are trasposed
            patch_locations = CSVDataset(locations,
                                col_groups={"image": "image", "location": ["y_min", "x_min"]},
                            )

            dataset = PatchWSIDataset(
                data=patch_locations,
                patch_size=patch_size,
                patch_level=patch_level,
                include_label=False,
                center_location=False,
                transform = eval_transforms,
                reader = reader
            )

            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            tile_predictions = []

            since = time.time()

            with torch.no_grad():
                for inputs in tqdm(dataloader, disable=args.silent):
                    tile = inputs['image'].to(device)
                    tile_location = inputs['image'].meta['location'].numpy()
                    
                    output = trained_model(tile)

                    batch_prediction = torch.nn.functional.softmax(output, dim=1).cpu().data.numpy()
                    predictions = np.concatenate((tile_location, batch_prediction), axis=1)
                    for i in range(len(predictions)):
                        tile_predictions.append(predictions[i])

            columns = ['y_min', 'x_min'] + classes
            predictions = pd.DataFrame(tile_predictions, columns=columns)
            predictions['x_min'] = predictions['x_min'].astype(int)
            predictions['y_min'] = predictions['y_min'].astype(int)
            predictions['x_max'] = predictions['x_min'] + patch_size
            predictions['y_max'] = predictions['y_min'] + patch_size
            predictions = predictions.reindex(columns=['x_min', 'y_min', 'x_max', 'y_max'] + classes)
            predictions.to_csv(slide_output+'.csv', index=False)

            time_elapsed = time.time() - since
            print('Inference complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        positive_tiles = (predictions[ranked_class] > thresh).sum()
        if positive_tiles >= hcp_threshold:
            algorithm_result = 3
        elif (positive_tiles < hcp_threshold) & (positive_tiles > lcp_threshold):
            algorithm_result = 2
        else:
            algorithm_result = 1

        annotation_file = None
        if args.xml or args.json:
            positive = predictions[predictions[ranked_class] > thresh]

            if len(positive) == 0:
                print(f'No annotations found at current threshold for {slide_name}')
            else:
                if args.xml:
                    annotation_file = slide_name+f'_{args.stain}'+'.xml'
                    annotation_path = os.path.join(directories['annotation_dir'], annotation_file)
                    if not os.path.exists(annotation_path):
                        # Make ASAP file
                        xml_header = """<?xml version="1.0"?><ASAP_Annotations>\t<Annotations>\n"""
                        xml_tail =  f"""\t</Annotations>\t<AnnotationGroups>\t\t<Group Name="{ranked_class}" PartOfGroup="None" Color="#64FE2E">\t\t\t<Attributes />\t\t</Group>\t</AnnotationGroups></ASAP_Annotations>\n"""

                        xml_annotations = ""
                        for index, tile in positive.iterrows():
                            xml_annotations = (xml_annotations +
                                                "\t\t<Annotation Name=\""+str(tile[ranked_class+'_probability'])+"\" Type=\"Polygon\" PartOfGroup=\""+ranked_class+"\" Color=\"#F4FA58\">\n" +
                                                "\t\t\t<Coordinates>\n" +
                                                "\t\t\t\t<Coordinate Order=\"0\" X=\""+str(tile['x_min'])+"\" Y=\""+str(tile['y_min'])+"\" />\n" +
                                                "\t\t\t\t<Coordinate Order=\"1\" X=\""+str(tile['x_max'])+"\" Y=\""+str(tile['y_min'])+"\" />\n" +
                                                "\t\t\t\t<Coordinate Order=\"2\" X=\""+str(tile['x_max'])+"\" Y=\""+str(tile['y_max'])+"\" />\n" +
                                                "\t\t\t\t<Coordinate Order=\"3\" X=\""+str(tile['x_min'])+"\" Y=\""+str(tile['y_max'])+"\" />\n" +
                                                "\t\t\t</Coordinates>\n" +
                                                "\t\t</Annotation>\n")
                        print('Creating automated annotation file for '+ slide_name)
                        with open(annotation_path, "w") as f:
                            f.write(xml_header + xml_annotations + xml_tail)
                    else:
                        print(f'Automated xml annotation file for {annotation_file} already exists.')

                if args.json:
                    annotation_file = slide_name+f'_{args.stain}'+'.geojson'
                    annotation_path = os.path.join(directories['annotation_dir'], annotation_file)
                    if not os.path.exists(annotation_path):
                        json_annotations = {"type": "FeatureCollection", "features":[]}
                        for index, tile in positive.iterrows():
                            color = [0, 0, 255]
                            status = str(ranked_class)

                            json_annotations['features'].append({
                                "type": "Feature",
                                "id": "PathDetectionObject",
                                "geometry": {
                                "type": "Polygon",
                                "coordinates": [
                                        [
                                            [tile['x_min'], tile['y_min']],
                                            [tile['x_max'], tile['y_min']],
                                            [tile['x_max'], tile['y_max']],
                                            [tile['x_min'], tile['y_max']],        
                                            [tile['x_min'], tile['y_min']]
                                        ]   
                                    ]
                                },
                                "properties": {
                                    "objectType": "annotation",
                                    "name": str(status)+'_'+str(round(tile[ranked_class], 4))+'_'+str(tile['x_min']) +'_'+str(tile['y_min']),
                                    "classification": {
                                        "name": status,
                                        "color": color
                                    }
                                }
                            })
                        print('Creating automated annotation file for ' + slide_name)
                        with open(annotation_path, "w") as f:
                            geojson.dump(json_annotations, f, indent=0)
                    else:
                        print(f'Automated geojson annotation file for {annotation_file} already exists')
        
        record = {
            'algorithm_cyted_sample_id': row['CYT ID'], 
            'slide_filename': row['slide_id'],
            'positive_tiles': positive_tiles,
            'algorithm_result': algorithm_result,
            'tile_mapping': annotation_file,
            'algorithm_version': f'{args.model_path.split("/")[-1]}',
            'redcap_repeat_instance': '1'
        }
        records.append(record)

    df = pd.DataFrame.from_dict(records)
    df.to_csv(os.path.join(directories['save_dir'], 'process_list.csv'), index=False)
    print(f'Number of HCP slides: {(df["algorithm_result"] == 3).sum()}')
    print(f'Number of LCP slides: {(df["algorithm_result"] == 2).sum()}')
    print(f'Number of Negative slides: {(df["algorithm_result"] == 1).sum()}')
