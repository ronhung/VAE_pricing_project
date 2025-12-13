import QuantLib as ql
import numpy as np
from american_put_pricer import read_vol_data

def price_asian_option_multi_KT(quote_date, vol_surface, K_grid, T_grid, eval_KTs, S0):
    """
    input: 1. (quote date, vol_surface), n (K,T)
    output: n NPV of Asian options
    """
    r = 0.02
    q = 0.0

    today = ql.Date(*map(int, quote_date.split("-")[::-1]))
    ql.Settings.instance().evaluationDate = today
    calendar = ql.NullCalendar()
    dayCounter = ql.Actual365Fixed()

    T_grid_expiry_dates = [today + int(T * 365 + 0.5) for T in T_grid]
    exact_ts = [dayCounter.yearFraction(today, d) for d in T_grid_expiry_dates]

    volMatrix = ql.Matrix(len(K_grid), len(T_grid))
    for i in range(len(K_grid)):
        for j in range(len(T_grid)):
            volMatrix[i][j] = vol_surface[i, j]

    # Adjust volMatrix
    for i in range(len(K_grid)):
        for j in range(len(T_grid)):
            if exact_ts[j] > 0:
                adjustment_factor = np.sqrt(T_grid[j] / exact_ts[j])
                volMatrix[i][j] *= adjustment_factor
            else:
                volMatrix[i][j] = 0.0

    # !!! 修正 1: 還原真實 Strike 用於構建波動率曲面 !!!
    # 原本: Abs_K_grid = [k * S0 for k in K_grid] -> 錯誤，k是log-moneyness，會導致負值
    Abs_K_grid = [S0 * k for k in K_grid]

    BlackSurf = ql.BlackVarianceSurface(today, calendar, T_grid_expiry_dates, Abs_K_grid, volMatrix, dayCounter)

    volTS = ql.BlackVolTermStructureHandle(BlackSurf)
    volTS.enableExtrapolation()

    spot = ql.QuoteHandle(ql.SimpleQuote(S0))
    ratesTS = ql.YieldTermStructureHandle(ql.FlatForward(today, r, dayCounter))
    divTS = ql.YieldTermStructureHandle(ql.FlatForward(today, q, dayCounter))

    process = ql.BlackScholesMertonProcess(spot, divTS, ratesTS, volTS)
    engine = ql.FdBlackScholesAsianEngine(process, tGrid=400, xGrid=400, aGrid=200)

    AsianC_NPVs = []
    AsianP_NPVs = []
    for i in range(len(eval_KTs)):
        # eval_KTs 中的 K 是 Log-Moneyness
        Moneyness = eval_KTs[i][0]
        T = eval_KTs[i][1]
        
        # !!! 修正 2: 計算真實 Strike 用於定價 !!!
        # 原本: Abs_K = Moneyness * S0 -> 錯誤
        Abs_K = S0 * Moneyness

        maturity = today + int(T * 365 + 0.5)
        average_type = ql.Average.Arithmetic
        exercise = ql.EuropeanExercise(maturity)
        pastFixings = 0
        asianFixingDates = [today + x for x in range(1, int(T * 365 + 1))]
        
        call_payoff = ql.PlainVanillaPayoff(ql.Option.Call, Abs_K)
        call_option = ql.DiscreteAveragingAsianOption(average_type, 0.0, pastFixings, asianFixingDates, call_payoff, exercise)
        call_option.setPricingEngine(engine)
        
        put_payoff = ql.PlainVanillaPayoff(ql.Option.Put, Abs_K)
        put_option = ql.DiscreteAveragingAsianOption(average_type, 0.0, pastFixings, asianFixingDates, put_payoff, exercise)
        put_option.setPricingEngine(engine)

        # 加入錯誤處理，避免因為極端值導致計算崩潰
        try:
            call_npv = call_option.NPV()
            put_npv = put_option.NPV()
        except RuntimeError:
            call_npv = 0.0
            put_npv = 0.0

        AsianC_NPVs.append(call_npv)
        AsianP_NPVs.append(put_npv)

    return np.array(AsianC_NPVs), np.array(AsianP_NPVs)


def generate_AsianOption_data_set(folder, N_data, vol_data_path, label, dataset_type="train"):
    # 1. read all vol data
    data, quote_dates, vol_surfaces, K_grid, T_grid, S0s = read_vol_data(vol_data_path, label)

    # 2. n (K,T) to eval for each quote date
    n = int(N_data / len(quote_dates))
    n_per_date = [n] * len(quote_dates)
    remainder = N_data % len(quote_dates)
    for i in range(remainder):
        n_per_date[i] += 1
    
    all_AsianC_NPVS_data = {"quote_date": [], "vol_surface": [], "K": [], "T": [], "NPV": [], "S0": []}
    all_AsianP_NPVS_data = {"quote_date": [], "vol_surface": [], "K": [], "T": [], "NPV": [], "S0": []}
    arb_date = []

    np.random.seed(42)

    for i in range(len(quote_dates)):
        K_min, K_max = min(K_grid), max(K_grid)
        T_min, T_max = min(T_grid), max(T_grid)

        eval_Ks = np.random.uniform(K_min, K_max, size=n_per_date[i])
        eval_Ts = np.random.uniform(T_min, T_max, size=n_per_date[i])

        eval_KTs = [[K, T] for K, T in zip(eval_Ks, eval_Ts)]
        if n_per_date[i] == 0:
            continue
        print(f"Evaluating {n_per_date[i]} (K,T) for quote date {quote_dates[i]}")
        S0_for_date = S0s[i]

        try:
            AsianC_NPVs, AsianP_NPVs = price_asian_option_multi_KT(quote_dates[i], vol_surfaces[i], K_grid, T_grid, eval_KTs, S0_for_date)
        except Exception as e:
            print(f"Error pricing Asian options for quote date {quote_dates[i]}: {e}")
            arb_date.append(quote_dates[i])
            continue

        for j in range(len(eval_KTs)):
            # !!! 修正 3: 儲存真實 Strike 而不是 Log-Moneyness !!!
            Moneyness = eval_KTs[j][0]
            Real_Strike = S0_for_date * Moneyness
            
            all_AsianC_NPVS_data["quote_date"].append(quote_dates[i])
            all_AsianC_NPVS_data["vol_surface"].append(vol_surfaces[i])
            all_AsianC_NPVS_data["K"].append(Real_Strike)  # 存入真實 Strike
            all_AsianC_NPVS_data["T"].append(eval_KTs[j][1])
            all_AsianC_NPVS_data["NPV"].append(AsianC_NPVs[j])
            all_AsianC_NPVS_data["S0"].append(S0_for_date)

            all_AsianP_NPVS_data["quote_date"].append(quote_dates[i])
            all_AsianP_NPVS_data["vol_surface"].append(vol_surfaces[i])
            all_AsianP_NPVS_data["K"].append(Real_Strike)  # 存入真實 Strike
            all_AsianP_NPVS_data["T"].append(eval_KTs[j][1])
            all_AsianP_NPVS_data["NPV"].append(AsianP_NPVs[j])
            all_AsianP_NPVS_data["S0"].append(S0_for_date)

    # 3. save data
    np.savez(
        f"{folder}/AsianCall_pricing_data_{dataset_type}.npz",
        quote_dates=all_AsianC_NPVS_data["quote_date"],
        vol_surfaces=all_AsianC_NPVS_data["vol_surface"],
        K=all_AsianC_NPVS_data["K"],
        T=all_AsianC_NPVS_data["T"],
        NPV=all_AsianC_NPVS_data["NPV"],
        UNDERLYING_LAST=all_AsianC_NPVS_data["S0"],
    )
    print(f"Asian Call data saved to {folder}/AsianCall_pricing_data_{dataset_type}.npz")

    np.savez(
        f"{folder}/AsianPut_pricing_data_{dataset_type}.npz",
        quote_dates=all_AsianP_NPVS_data["quote_date"],
        vol_surfaces=all_AsianP_NPVS_data["vol_surface"],
        K=all_AsianP_NPVS_data["K"],
        T=all_AsianP_NPVS_data["T"],
        NPV=all_AsianP_NPVS_data["NPV"],
        UNDERLYING_LAST=all_AsianP_NPVS_data["S0"],
    )
    print(f"Asian Put data saved to {folder}/AsianPut_pricing_data_{dataset_type}.npz")