import pdfkit
import glob
import os
import json
import argparse
import sys

def convert_html_to_pdf(html_path, pdf_save_path):
    try:
        pdfkit.from_file(html_path, pdf_save_path)
    #It might throw an OSError. But the conversion is complete irrespective.
    except OSError:
        pass

def main(args):
    with open(args.config_path) as json_file:
        config_dict = json.load(json_file)
    #Get directory paths for all symbols as a list
    symbol_paths = [folder for folder in glob.glob(os.path.join(config_dict['annual_reports_html_save_directory'], '*')) \
                        if os.path.isdir(folder)]
    for i, symbol_path in enumerate(symbol_paths):
        #Get symbol name
        symbol = symbol_path.split('/')[-1]
        #Get annual_report dates for the symbol. Directories are named by the annual_report date
        ar_dates_symbol_paths = [folder for folder in glob.glob(os.path.join(symbol_path, '*')) \
                                    if os.path.isdir(folder)]
        #Iterate over each date and convert the html file to pdf file
        for ar_dates_symbol_path in ar_dates_symbol_paths:
            ar_paths = [file for file in glob.glob(os.path.join(ar_dates_symbol_path, '*')) \
                            if os.path.isfile(file)]
            #ar_paths should be a list of 1 element only i.e the report
            assert len(ar_paths)==1
            date = ar_dates_symbol_path.split('/')[-1]
            pdf_save_dir = os.path.join(config_dict['annual_reports_pdf_save_directory'], symbol, date)
            pdf_save_path = os.path.join(pdf_save_dir, date+'.pdf')
            #If path exists, then the conversion has already happened before
            if os.path.exists(pdf_save_path):
                continue
            else:
                if not os.path.exists(pdf_save_dir):
                    os.makedirs(pdf_save_dir)
                convert_html_to_pdf(ar_paths[0], pdf_save_path)
        print('Completed: {}/{}'.format(i+1, len(symbol_paths)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', dest='config_path', type=str,
                        required=True,
                        help='''Full path of config.json''')
    main(args=parser.parse_args())
    sys.exit(0)