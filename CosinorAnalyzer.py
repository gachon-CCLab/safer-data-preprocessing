import math
import warnings
import datetime
import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from tqdm import tqdm  # 진행 표시를 위해 추가

class CosinorAnalyzer:
    def __init__(self, 
                 data: pd.DataFrame, 
                 window_length_hours: float,
                 coverage_threshold: float = 0.5,
                 sensor: str = 'ENMO',
                 name_col: str = '이름'):
        """
        개선된 cosinor 분석기:

        - 데이터 전체 구간에 대해 window를 이동시키며 cosinor 분석 수행.
        - 위상 언래핑 없이 phase를 0~24시간 범위로 변환.
        - 커버리지 임계값, 샘플링 레이트 자동 계산.
        - 주기는 24시간 고정.
        - sensor 인자를 통해 분석 대상 컬럼 결정.
          예: sensor='ENMO' -> 'ENMO_mean' 분석
              sensor='HR'   -> 'HR_mean'   분석
        - 환자 이름 기준으로 개별적으로 분석 수행.

        Parameters
        ----------
        data : pd.DataFrame
            time (pd.Timestamp), ENMO_mean 또는 HR_mean, nonwearing (bool), 이름 (str) 열 등을 포함.
        window_length_hours : float
            분석 윈도우 길이(시간)
        coverage_threshold : float
            데이터 커버리지 최소 비율(0~1), 기본값 0.5
        sensor : str
            분석할 센서명. 예: 'ENMO', 'HR'
        name_col : str
            환자 이름을 식별하는 컬럼명. 기본값은 '이름'.
        """
        self.data = data.copy()
        self.window_length_hours = window_length_hours
        self.coverage_threshold = coverage_threshold
        self.period = 24.0  # 고정
        self.sensor = sensor
        self.name_col = name_col
        # 센서에 따른 컬럼명 결정
        self.sensor_col = f"{self.sensor}_mean"

        # 필수 컬럼 확인
        required_cols = ['targetTime', self.sensor_col, 'nonwearing', self.name_col]
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Data must contain '{col}' column.")

        # time 컬럼이 datetime인지 확인
        if not pd.api.types.is_datetime64_any_dtype(self.data['targetTime']):
            raise ValueError("data['time'] must be a datetime type.")

        # 시간 순 정렬 보장
        self.data.sort_values(['이름', 'targetTime'], inplace=True)
        self.data.reset_index(drop=True, inplace=True)

        # 샘플링 레이트 추정 (분 단위) - 각 환자별로 확인
        # 여기서는 전체 데이터의 샘플링 레이트를 사용합니다.
        # 필요시 환자별로 다르게 설정할 수 있습니다.
        time_diffs = self.data.groupby(self.name_col)['targetTime'].diff().dropna()
        if len(time_diffs) == 0:
            raise ValueError("Not enough data points to determine sampling rate.")
        
        median_diff = time_diffs.median().total_seconds() / 60.0  # 분 단위
        self.sampling_interval_minutes = median_diff

    # def _prepare_window_data(self, df_group, ref_time: pd.Timestamp, global_reference_time):
    #     """
    #     ref_time을 기준으로 window_length_hours만큼 과거 데이터 추출
    #     """
    #     start_time = ref_time - pd.Timedelta(hours=self.window_length_hours)
    #     end_time = ref_time

    #     df_window = df_group[(df_group['time'] >= start_time) & (df_group['time'] <= end_time)]
    #     df_window = df_window[df_window['nonwearing'] == False]

    #     # 예상 포인트 수 계산
    #     expected_points = int((self.window_length_hours * 60) / self.sampling_interval_minutes)
    #     actual_points = len(df_window)

    #     if actual_points == 0:
    #         return None

    #     coverage_ratio = actual_points / expected_points
    #     if coverage_ratio < self.coverage_threshold:
    #         return None

    #     # t_hours 컬럼 추가
    #     df_window = df_window.copy()
    #     df_window['t_hours'] = (df_window['time'] - global_reference_time).dt.total_seconds() / 3600.0

        # return df_window

    def _prepare_window_data(self, df_group, ref_time: pd.Timestamp, global_reference_time):
        """
        ref_time 이전 window_length_hours만큼 과거 데이터 추출
        """
        start_time = ref_time - pd.Timedelta(hours=self.window_length_hours)
        end_time = ref_time - pd.Timedelta(seconds=1)  # ref_time 이전까지만 사용

        df_window = df_group[(df_group['targetTime'] >= start_time) & (df_group['targetTime'] <= end_time)]
        df_window = df_window[df_window['nonwearing'] == False]

        # 예상 포인트 수 계산
        expected_points = int((self.window_length_hours * 60) / self.sampling_interval_minutes)
        actual_points = len(df_window)

        # t_hours 컬럼 추가 (비어있더라도 추가 필요)
        df_window = df_window.copy()
        df_window['t_hours'] = (df_window['targetTime'] - global_reference_time).dt.total_seconds() / 3600.0

        return df_window, actual_points, expected_points

    def _perform_cosinor_analysis(self, df_window):
        # sensor에 따른 컬럼명 사용
        if self.sensor_col not in df_window.columns:
            warnings.warn(f"{self.sensor_col} column not found in the data window.")
            return None

        y = df_window[self.sensor_col].values
        t = df_window['t_hours'].values

        omega = 2.0 * np.pi / self.period
        X = np.column_stack([np.ones_like(t), np.cos(omega * t), np.sin(omega * t)])

        try:
            params, residuals, rank, s = lstsq(X, y, rcond=None)
        except Exception as e:
            warnings.warn(f"An error occurred during cosinor analysis: {e}. Returning default values.")
            return None

        M, A_coef, B_coef = params
        amplitude = np.sqrt(A_coef**2 + B_coef**2)

        # 위상을 0~24시간 범위로 변환
        phase_radian = math.atan2(B_coef, A_coef)  # -pi ~ pi
        phase_hours = (phase_radian / (2.0 * np.pi)) * self.period
        if phase_hours < 0:
            phase_hours += self.period  # 0 ~ 24 시간으로 변환

        return M, amplitude, phase_hours

    def run(self):
        """
        전체 데이터 구간에 대해 환자별로 window를 이동하며 cosinor 분석 수행.
        결과를 pandas DataFrame으로 반환.

        Returns
        -------
        pd.DataFrame
            '이름', 'time', 'MESOR', 'Amplitude', 'Phase_hours' 컬럼을 포함.
        """
        results = []
        grouped = self.data.groupby(self.name_col)

        for name, group in tqdm(grouped, desc="Processing patients"):
            # 글로벌 기준 시간 설정 (각 환자별로 최소 시간)
            global_reference_time = group['targetTime'].min()

            # 시작 분석 시간 설정
            start_analysis_time = global_reference_time + pd.Timedelta(hours=self.window_length_hours)
            end_analysis_time = group['targetTime'].max()

            # 유효한 reference 시간 추출
            valid_times = group[group['targetTime'] >= start_analysis_time]['targetTime']

            for ref_time in valid_times:
                df_window, actual_points, expected_points = self._prepare_window_data(group, ref_time, global_reference_time)

                if actual_points == 0:
                    # 완전 비어있으면 NaN
                    M, amplitude, phase = (np.nan, np.nan, np.nan)
                elif (actual_points / expected_points) < self.coverage_threshold:
                    # 커버리지 부족하면 zero-padding 결과
                    M, amplitude, phase = (0.0, 0.0, 0.0)
                else:
                    cosinor_result = self._perform_cosinor_analysis(df_window)
                    if cosinor_result is None:
                        M, amplitude, phase = (np.nan, np.nan, np.nan)
                    else:
                        M, amplitude, phase = cosinor_result

                results.append({
                    self.name_col: name,
                    'time': ref_time,
                    'MESOR': M,
                    'Amplitude': amplitude,
                    'Phase_hours': phase
                })

        result_df = pd.DataFrame(results)
        return result_df
