import pandas as pd

# Load the analysis results
df = pd.read_csv('group_analysis_outputs/complete_group_analysis.csv')

print('COMPREHENSIVE GROUP ANALYSIS SUMMARY')
print('=' * 60)
print(f'Total analyses completed: {len(df)}')
print(f'Successful analyses: {len(df[df.analysis_status == "SUCCESS"])}')
print()

print('RESULTS BY GROUP:')
print('-' * 40)
for group in sorted(df.group.unique()):
    group_data = df[df.group == group]
    successful = group_data[group_data.analysis_status == 'SUCCESS']
    quality_good = successful[successful.overall_quality_good == True]
    
    print(f'{group} GROUP:')
    print(f'  Total analyses: {len(group_data)}')
    print(f'  Successful: {len(successful)}')
    print(f'  Good quality: {len(quality_good)}/{len(successful)} ({len(quality_good)/len(successful)*100:.1f}%)')
    
    if len(successful) > 0:
        snr_data = successful[successful.snr.notna()]
        vol_data = successful[successful.brain_volume_ml.notna()]
        cnr_data = successful[successful.cnr.notna()]
        
        print(f'  SNR: {snr_data.snr.mean():.0f} ± {snr_data.snr.std():.0f}')
        print(f'  CNR: {cnr_data.cnr.mean():.0f} ± {cnr_data.cnr.std():.0f}')
        print(f'  Volume: {vol_data.brain_volume_ml.mean():.0f} ± {vol_data.brain_volume_ml.std():.0f} ml')
        print(f'  Quality Score: {successful.quality_score.mean():.2f} ± {successful.quality_score.std():.2f}')
    print()

print('DEMOGRAPHICS SUMMARY:')
print('-' * 30)
# Load demographics
import os
import openpyxl

excel_path = r"D:\data_NIMHANS\age_gender.xlsx"
if os.path.exists(excel_path):
    demo_df = pd.read_excel(excel_path)
    for group in ['HC', 'PIGD', 'TDPD']:
        group_demo = demo_df[demo_df[group] == 1]
        male_count = len(group_demo[group_demo.gender == 1])  # 1 = Male  
        female_count = len(group_demo[group_demo.gender == 2])  # 2 = Female
        print(f'{group}: Age {group_demo.age_assessment.mean():.1f}±{group_demo.age_assessment.std():.1f}, M/F: {male_count}/{female_count}')

print()
print('ANALYSIS COMPLETE!')
print(f'Results saved to: group_analysis_outputs/complete_group_analysis.csv')
