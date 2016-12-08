import pandas as pd

def preprocess_student_alcohol_file(filename):
    df = pd.read_csv(filename, sep=';')

    print 'Sample records : \n', df.head()

    df_walc = df.drop('Dalc', axis=1, inplace=False)
    df_walc['walc'] = df_walc['Walc']
    df_walc.drop('Walc', axis=1, inplace=True)
    df_walc.to_csv('student-mat-walc.csv', index=False, sep=',', header=False)

    df_dalc = df.drop('Walc', axis = 1, inplace = False)
    df_dalc['dalc'] = df_dalc['Dalc']
    df_dalc.drop('Dalc', axis=1, inplace=True)
    df_dalc.to_csv('student-mat-dalc.csv', index=False, sep=',', header=False)


if __name__ == '__main__':
    print 'This is my Random row selection program'
    preprocess_student_alcohol_file('student-mat.csv')
