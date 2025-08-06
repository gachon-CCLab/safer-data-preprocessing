import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter

class LocationProcessor:
    @staticmethod
    def load_data_from_csv(file_path):
        """
        CSV 파일을 로드하여 전처리하는 함수.
        :param file_path: CSV 파일 경로
        :return: 전처리된 DataFrame
        """
        try:
            data = pd.read_csv(file_path, index_col=False, encoding='utf-8')
            print(f"CSV 파일 {file_path}이 성공적으로 로드되었습니다.")
            
            data = LocationProcessor.preprocess_location_data(data)
            data = LocationProcessor.resample_and_calculate(data)
            
            print("데이터 전처리가 완료되었습니다.")
            return data
        except Exception as e:
            print(f"파일 로드 중 오류 발생: {e}")
        return None
    
    @staticmethod
    def preprocess_location_data(data):
        """
        위치 데이터를 전처리하는 함수.
        'targetTime'을 datetime으로 변환하고, 인덱스를 설정합니다.
        :param data: 위치 데이터 DataFrame
        :return: 전처리된 DataFrame
        """
        try:
            data['targetTime'] = pd.to_datetime(data['targetTime'])
            data.set_index(['이름', 'targetTime'], inplace=True)
            
            print("위치 데이터 전처리가 완료되었습니다.")
            return data
        except Exception as e:
            print(f"위치 데이터 전처리 중 오류 발생: {e}")
        return data
    
    @staticmethod
    def calculate_mode(df):
        """
        주어진 DataFrame의 최빈값을 계산하는 함수.
        :param df: DataFrame
        :return: 최빈값 Series
        """
        modes = df.mode()
        if not modes.empty:
            return modes.iloc[0]
        else:
            return None
    
    @staticmethod
    def calculate_entropy_with_padding(group, min_hours=24):
        """
        한 환자 그룹에서 좌표 분포 엔트로피를 계산 (zero-padding 적용)
        :param group: 위치 데이터 그룹
        :param min_hours: 최소 필요 시간 (기본 24시간)
        :return: 엔트로피, 정규화된 엔트로피
        """
        if group.empty:
            return 0.0, 0.0
        
        # 데이터가 최소 시간보다 적으면 zero-padding
        time_span = (group['targetTime'].max() - group['targetTime'].min()).total_seconds() / 3600
        if time_span < min_hours:
            # zero-padding: 부족한 시간만큼 기본 위치(0,0) 추가
            padding_count = int((min_hours - time_span) * 4)  # 15분 간격으로 계산
            loc_series = [(0.0, 0.0)] * padding_count
        else:
            loc_series = []
        
        # 실제 데이터 추가
        real_locs = list(zip(group['위도'].round(5), group['경도'].round(5)))
        loc_series.extend(real_locs)
        
        if not loc_series:
            return 0.0, 0.0
        
        counts = np.array(list(Counter(loc_series).values()), dtype=float)
        probs = counts / counts.sum()
        
        # 엔트로피 계산
        entropy = -(probs * np.log(probs + 1e-10)).sum()
        max_entropy = np.log(len(probs)) if len(probs) > 1 else 1
        return entropy, entropy / max_entropy
    
    @staticmethod
    def calculate_entropy(group):
        """
        기존 엔트로피 계산 함수 (호환성 유지)
        """
        return LocationProcessor.calculate_entropy_with_padding(group, min_hours=24)
    
    @staticmethod
    def calculate_location_variance(group):
        """
        위치 데이터의 위도 및 경도에 대한 분산을 계산하는 함수.
        :param group: DataFrame 그룹
        :return: 위도 및 경도 분산의 로그 값
        """
        lat_var = np.var(group['위도'].astype(float))
        lon_var = np.var(group['경도'].astype(float))
        return np.log(lat_var + lon_var + 1e-10)
    
    @staticmethod
    def sliding_window_variability(df, window_size):
        """
        슬라이딩 윈도우를 이용한 위치 가변성 계산
        """
        variability = []
        
        # 데이터가 윈도우 크기보다 작은 경우 처리
        if len(df) < window_size:
            if len(df) > 1:
                # 전체 데이터로 가변성 계산
                var = LocationProcessor.calculate_location_variance(df)
                variability.append(var)
            else:
                # 데이터가 1개 이하면 가변성 0
                variability.append(0.0)
            return variability
        
        # 슬라이딩 윈도우 계산 (겹치지 않는 윈도우)
        for start in range(0, len(df), window_size):
            end = min(start + window_size, len(df))
            window = df.iloc[start:end]
            
            if len(window) > 1:
                var = LocationProcessor.calculate_location_variance(window)
                variability.append(var)
            else:
                # 윈도우 크기가 1 이하면 가변성 0
                variability.append(0.0)
        
        return variability
    
    @staticmethod
    def resample_and_calculate(data):
        """
        데이터를 리샘플링하고, 엔트로피 및 위치 가변성을 계산하는 함수.
        zero-padding 로직이 적용된 버전
        :param data: 전처리된 위치 데이터 DataFrame
        :return: 리샘플링 및 계산된 엔트로피 및 위치 가변성을 포함한 DataFrame
        """
        try:
            df_minute_mode = data.groupby(level='이름').resample('15T', level='targetTime').apply(LocationProcessor.calculate_mode)
            df_minute_mode = df_minute_mode.reset_index()
            df_minute_mode['Date'] = df_minute_mode['targetTime'].dt.date
            
            # 24시간 엔트로피 계산 (zero-padding 적용)
            daily_entropy_df = data.reset_index()
            daily_entropy_df['Date'] = daily_entropy_df['targetTime'].dt.date
            
            # 각 날짜별로 24시간 이전 데이터까지 포함하여 계산
            daily_entropy_results = []
            for date in daily_entropy_df['Date'].unique():
                date_data = daily_entropy_df[daily_entropy_df['Date'] <= date]
                # 24시간 이내 데이터만 사용
                cutoff_time = pd.to_datetime(date) + pd.Timedelta(days=1)
                date_data = date_data[date_data['targetTime'] >= cutoff_time - pd.Timedelta(hours=24)]
                
                entropy, norm_entropy = LocationProcessor.calculate_entropy_with_padding(date_data, min_hours=24)
                daily_entropy_results.append({
                    'Date': date,
                    'Daily_Entropy': entropy,
                    'Normalized_Daily_Entropy': norm_entropy
                })
            
            daily_entropy_result = pd.DataFrame(daily_entropy_results)
            df_minute_mode = df_minute_mode.merge(daily_entropy_result, on='Date', how='left')
            
            
            # 위치 가변성 계산 - 6시간 이상의 데이터가 있는 경우에만 적용
            location_variability = LocationProcessor.sliding_window_variability(df_minute_mode, 8)
            
            # Location_Variability 컬럼을 전체 DataFrame 크기에 맞게 초기화
            df_minute_mode['Location_Variability'] = np.nan
            
            # 계산된 값들을 해당 인덱스에 할당
            if location_variability:
                # 슬라이딩 윈도우의 결과를 적절한 위치에 할당
                for i, var_value in enumerate(location_variability):
                    start_idx = i * 8  # 윈도우 크기만큼 이동
                    end_idx = min(start_idx + 8, len(df_minute_mode))
                    # 해당 윈도우 구간의 모든 행에 같은 값 할당
                    df_minute_mode.iloc[start_idx:end_idx, df_minute_mode.columns.get_loc('Location_Variability')] = var_value
            
            # NaN 값을 0으로 채우기 (zero-padding 효과)
            entropy_columns = ['Daily_Entropy', 'Normalized_Daily_Entropy', 'Eight_Hour_Entropy', 'Normalized_Eight_Hour_Entropy', 'Location_Variability']
            df_minute_mode[entropy_columns] = df_minute_mode[entropy_columns].fillna(0)
            
            print("리샘플링 및 엔트로피 계산이 완료되었습니다 (zero-padding 적용).")
            return df_minute_mode
        except Exception as e:
            print(f"리샘플링 및 계산 중 오류 발생: {e}")
        return data
    
    @staticmethod
    def assign_location_labels(data, location_dict):
        """
        위치 데이터를 기준으로 가장 가까운 위치 레이블을 할당하는 함수.
        :param data: 위치 데이터 DataFrame
        :param location_dict: 위치와 좌표를 매핑하는 딕셔너리
        :return: 위치 레이블이 할당된 DataFrame
        """
        try:
            data = data.fillna(0)
            data['위도'] = pd.to_numeric(data['위도'], errors='coerce')
            data['경도'] = pd.to_numeric(data['경도'], errors='coerce')
            data = data.fillna(0)

            coords = np.array(list(location_dict.keys()))
            neigh = NearestNeighbors(n_neighbors=1)
            neigh.fit(coords)

            def get_nearest_location(row):
                distance, index = neigh.kneighbors([[row['위도'], row['경도']]])
                nearest_coord = coords[index[0][0]]
                return location_dict[tuple(nearest_coord)]

            data['place'] = data.apply(get_nearest_location, axis=1)
            data = data.drop(columns=['위도', '경도'])

            print("위치 레이블 할당이 완료되었습니다.")
            return data
        except Exception as e:
            print(f"위치 레이블 할당 중 오류 발생: {e}")
        return data