import pandas as pd
import os

excel_path = r'D:\data_NIMHANS\age_gender.xlsx'
if os.path.exists(excel_path):
    demo_df = pd.read_excel(excel_path)
    print('DEMOGRAPHICS SUMMARY:')
    print('-' * 30)
    for group in ['HC', 'PIGD', 'TDPD']:
        group_demo = demo_df[demo_df[group] == 1]
        male_count = len(group_demo[group_demo.gender == 1])  # 1 = Male
        female_count = len(group_demo[group_demo.gender == 2])  # 2 = Female
        print(f'{group}: n={len(group_demo)}, Age {group_demo.age_assessment.mean():.1f}±{group_demo.age_assessment.std():.1f}, M/F: {male_count}/{female_count}')
    
    print()
    print('OVERALL DEMOGRAPHICS:')
    print(f'Total subjects: {len(demo_df)}')
    print(f'Age range: {demo_df.age_assessment.min():.0f}-{demo_df.age_assessment.max():.0f} years')
    print(f'Overall age: {demo_df.age_assessment.mean():.1f}±{demo_df.age_assessment.std():.1f} years')
    total_male = len(demo_df[demo_df.gender == 1])
    total_female = len(demo_df[demo_df.gender == 2])
    print(f'Overall M/F: {total_male}/{total_female}')
else:
    print(f'Demographics file not found at: {excel_path}')
    print('Please check the file path.')
