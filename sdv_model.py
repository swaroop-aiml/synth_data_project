from sdv.datasets.local import load_csvs
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer , TVAESynthesizer 
import pandas as pd
from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import get_column_plot

# datasets = load_csvs(
#     folder_name='data/',
#     read_csv_parameters={
#         'skipinitialspace': True,
        # 'encoding': 'utf_8'
    # }

real_data = pd.read_csv('data/bank-full.csv', delimiter=',')


# real_data = datasets['bank-full']
 # Keep as object if it cannot be converted
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)

synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(data=real_data)
synthesizer.save('./model/copula_synthesizer.pkl')


synthesizer_2 = TVAESynthesizer(metadata)
synthesizer_2.fit(data=real_data)
synthesizer_2.save('./model/tvae_synthesizer.pkl')


synthesizer_1 = CTGANSynthesizer(metadata)
synthesizer_1.fit(data=real_data)
synthesizer_1.save('./model/ctgan_synthesizer.pkl')


metadata

synthesizer = TVAESynthesizer.load('./model/tvae_synthesizer.pkl')
synthetic_data = synthesizer.sample(num_rows=100000)
quality = evaluate_quality(real_data, synthetic_data, metadata)

# from realtabformer import REaLTabFormer
# model = REaLTabFormer.load_from_dir(r'model/results/rtf_bank_model/id000017264749831707004928')


synthetic_data.dtypes
real_data.dtypes
