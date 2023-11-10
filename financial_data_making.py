import OpenDartReader
import dart_fss
import pandas as pd

# input year: 2022
# Result: year - 1 

class financial_dict():
    def __init__(self, api_key):
        self.dart = OpenDartReader(api_key)
        self.dict_1 = None
        self.dict_2 = None
                
    def dart_extract(self, code, year):
        # 매출액
        sales = self.dart.finstate(code, year)
        # real_sales = sales[(sales['fs_nm']=='연결재무제표')&(sales['account_nm']=='매출액')].iloc[0, :] # 매출액 

        c_sales = sales[(sales['fs_nm']=='연결재무제표')&(sales['account_nm']=='매출액')]['thstrm_amount'].values # 매출액
        p_sales = sales[(sales['fs_nm']=='연결재무제표')&(sales['account_nm']=='매출액')]['frmtrm_amount'].values # t-1기 매출액

        div_cash = self.dart.report(code, '배당', year) #(연결)현금배당성향(%)
        c_div = div_cash[div_cash['se']=='(연결)현금배당성향(%)']['thstrm'].values # 순이익이 마이너스일 경우 0% # 배당금 / 당기순이익
        p_div = div_cash[div_cash['se']=='(연결)현금배당성향(%)']['frmtrm'].values

        holder = self.dart.report(code, '최대주주', year) # 보통주 계 + 우선수 계 (%)
        ratio_stakeholder = holder[(holder['stock_knd']=='보통주')&(holder['nm']=='계')].iloc[0, -4]

        exec = self.dart.report(code, '임원', year) #'사외이사' / 전체 임원 수   
        out_exec = exec[exec['ofcps'] == '사외이사'].shape[0]
        exec = exec.shape[0]

        employee = self.dart.report(code, '직원', year)['rgllbr_co'].to_list() # 총 직원 수
        
        
        
        self.dict_1= {'sales': c_sales, 
                     'pre_sales': p_sales, 
                     'cash_div': c_div, 
                     'pre_cash_civ': p_div,
                     'stakeholder' : ratio_stakeholder,
                     'num_outexecutives': out_exec,
                     'num_executives': exec,
                     'num_employee': employee}
        
        return self.dict_1
    
    def html_extract(self, code, year):
        lst = self.dart.list(code, start= str(year), end=str(year+1))
        report_num = lst[lst['report_nm'].str.contains('사업보고서')].iloc[0, 5] # 해당년도 사업 보고서가 없는 경우
        (report_name, report_num) = lst[lst['report_nm'].str.contains('사업보고서')][['report_nm', 'rcept_no']].iloc[0].items()
        print(report_name[1])
        index = report_num[1]
        url = self.dart.sub_docs(index) #사업보고서
        i = url[url['title'].str.contains('연결재무제표')].index[0]
        html = url.loc[i, 'url']

        docu = pd.read_html(html)
        if docu[1].empty | docu[3].empty:
            print('Some document cannot be extracted')
        else:
            fin_state = docu[1].rename(columns={'Unnamed: 0': 'tag'}) # 재무상태표

            asset = fin_state[fin_state['tag'] == '자산총계'].iloc[0, 1] # 자산 총계
            debt= fin_state[fin_state['tag'] == '부채총계'].iloc[0, 1] # 부채 총계
            intangible = fin_state[fin_state['tag'] == '무형자산'].iloc[0, 1] # 무형자산 

            income = docu[3].rename(columns={'Unnamed: 0': 'tag'}) # 손익계산서

            profit = income[income['tag'].str.contains('영업이익')].iloc[0, 1] # ()는 마이너스를 뜻함 # 영업이익(손실)
            c_profit_tax = income[income['tag'].str.contains('법인세비용차감전')].iloc[0, 1] # t기 법인세비용차감전순이익(손실)
            p_profit_tax = income[income['tag'].str.contains('법인세비용차감전')].iloc[0, 2] # t-1기 법인세비용차감전순이익(손실)
            
            self.dict_2 = {'asset': asset,
                        'debt': debt,
                        'intang': intangible,
                        'profit': profit,
                        'profit_without_tax': c_profit_tax,
                        'pre_profit_without_tax': p_profit_tax}
        
        return self.dict_2
        