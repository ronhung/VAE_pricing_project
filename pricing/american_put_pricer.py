import QuantLib as ql
import numpy as np


def read_vol_data(path, label):
    """
    will return all vol data from the npz file
    """
    data = np.load(path)
    print("Keys in the loaded data:", data.keys())

    quote_dates = data["quote_dates"]
    k_grid = data["k_grid"]
    K_grid = np.exp(k_grid)
    T_grid = data["T_grid"]

    if label == "vol_" or label == "post_vol_":
        vol_surfaces = data["surfaces_grid"]
    elif label == "total_var_" or label == "post_total_var_":
        vol_surfaces = []
        for i in range(len(data["surfaces_grid"])):
            # convert vol surface to total variance surface
            total_var_grid = data["surfaces_grid"][i]
            vol_surface = np.sqrt(total_var_grid / np.maximum(T_grid, 1e-6))  # Convert total variance to vol surface
            vol_surfaces.append(vol_surface)
    else:
        raise ValueError("Unsupported label. Use 'vol_' or 'total_var_'.")

    S0s = data["S0s"] # <--- 1. 讀取 S0s
    return data, quote_dates, vol_surfaces, K_grid, T_grid, S0s # <--- 2. 回傳 S0s

def price_american_put_options_multi_KT(quote_date, vol_surface, K_grid, T_grid, eval_KTs, S0):
    """
    input: 1. (quote date, vol_surface), n (K,T)
    output: n NPV of american put options
    """
    # some constants
    # S0 由參數傳入
    r = 0.02
    q = 0.0
    
    # environment setup
    today = ql.Date(*map(int, quote_date.split("-")[::-1]))
    ql.Settings.instance().evaluationDate = today
    calendar = ql.NullCalendar()
    dayCounter = ql.Actual365Fixed()

    T_grid_expiry_dates = [today + int(T * 365 + 0.5) for T in T_grid]
    
    # build vol surface object
    volMatrix = ql.Matrix(len(K_grid), len(T_grid))
    for i in range(len(K_grid)):
        for j in range(len(T_grid)):
            volMatrix[i][j] = vol_surface[i, j]

    # !!! 修正點 1: Vol Surface 需要使用「絕對 Strike」來建立 !!!
    # K_grid 目前是 Moneyness (K/S)，需要乘上 S0 變回絕對價格
    Abs_K_grid = [k * S0 for k in K_grid] 

    BlackSurf = ql.BlackVarianceSurface(today, calendar, T_grid_expiry_dates, Abs_K_grid, volMatrix, dayCounter)

    volTS = ql.BlackVolTermStructureHandle(BlackSurf)
    volTS.enableExtrapolation()

    spot = ql.QuoteHandle(ql.SimpleQuote(S0))
    ratesTS = ql.YieldTermStructureHandle(ql.FlatForward(today, r, dayCounter))
    divTS = ql.YieldTermStructureHandle(ql.FlatForward(today, q, dayCounter))

    process = ql.BlackScholesMertonProcess(spot, divTS, ratesTS, volTS)
    engine = ql.FdBlackScholesVanillaEngine(process, 400, 400)

    AmericanP_NPVs = []
    for i in range(len(eval_KTs)):
        Moneyness = eval_KTs[i][0] # 這裡拿到的 K 其實是 Moneyness
        T = eval_KTs[i][1]
        
        # !!! 修正點 2: Option Payoff 需要使用「絕對 Strike」 !!!
        Abs_K = Moneyness * S0 
        
        maturity = today + int(T * 365 + 0.5)
        payoff = ql.PlainVanillaPayoff(ql.Option.Put, Abs_K) # 使用 Abs_K
        exercise = ql.AmericanExercise(today, maturity)
        option = ql.VanillaOption(payoff, exercise)
        option.setPricingEngine(engine)
        
        try:
            NPV = option.NPV()
        except RuntimeError:
             # 如果插值失敗，給一個合理的值或 NaN (視情況而定)
             NPV = 0.0 

        AmericanP_NPVs.append(NPV)

    return np.array(AmericanP_NPVs)

def generate_AmericanPut_data_set(folder, N_data, vol_data_path, label, dataset_type="test"):

    # 1. read all vol data
    data, quote_dates, vol_surfaces, K_grid, T_grid, S0s = read_vol_data(vol_data_path, label)
    """
    data prepared
    k_grid = np.linspace(-0.3, 0.3, 41)  <-- 這裡是 log-moneyness
    T_grid = np.linspace(0.05, 1.0, 20)
    """
    # 2. n (K,T) to eval for each quote date
    n = int(N_data / len(quote_dates))
    n_per_date = [n] * len(quote_dates)
    remainder = N_data % len(quote_dates)
    for i in range(remainder):
        n_per_date[i] += 1
    print("n_per_date", n_per_date)

    all_AmericanP_NPVS_data = {"quote_date": [], "vol_surface": [], "K": [], "T": [], "NPV": [], "S0": []}
    arb_date = []

    # Set random seed for reproducible sampling
    np.random.seed(42)

    # !!! 修正 1: 計算 Moneyness (exp空間) 的範圍 !!!
    # K_grid 是 log-moneyness (-0.3, 0.3)，我們要採樣的是 Moneyness (0.74, 1.35)
    K_min_exp = np.exp(np.min(K_grid))
    K_max_exp = np.exp(np.max(K_grid))
    
    # T 不需要轉換
    T_min, T_max = np.min(T_grid), np.max(T_grid)

    for i in range(len(quote_dates)):
        # !!! 修正 2: 使用 exp 轉換後的範圍採樣 !!!
        eval_Ks = np.random.uniform(K_min_exp, K_max_exp, size=n_per_date[i])
        eval_Ts = np.random.uniform(T_min, T_max, size=n_per_date[i])

        eval_KTs = [[K, T] for K, T in zip(eval_Ks, eval_Ts)]
        
        if n_per_date[i] == 0:
            continue
            
        # print(f"Evaluating {n_per_date[i]} (K,T) for quote date {quote_dates[i]}")
        S0_for_date = S0s[i]

        try:
            # 注意: 這裡傳入的 eval_KTs 中的 K 已經是 Moneyness (0.7~1.3) 了
            # 所以 price_american_put_options_multi_KT 裡的 Abs_K = Moneyness * S0 邏輯就會變正確
            AmericanP_NPVS = price_american_put_options_multi_KT(quote_dates[i], vol_surfaces[i], K_grid, T_grid, eval_KTs, S0_for_date)
        except Exception as e:
            print(f"Error processing quote date {quote_dates[i]}: {e}")
            arb_date.append(quote_dates[i])
            continue
        
        # print("AmericanP_NPVS", AmericanP_NPVS)
        
        for j in range(len(AmericanP_NPVS)):
            npv = AmericanP_NPVS[j]
            
            # !!! 修正 3: 資料清洗 - 過濾無效價格 !!!
            if np.isnan(npv) or np.isinf(npv) or npv < 0:
                # print(f"Warning: Invalid price (NPV={npv}) at index {j} for date {quote_dates[i]}, skipping.")
                continue 

            all_AmericanP_NPVS_data["quote_date"].append(quote_dates[i])
            all_AmericanP_NPVS_data["vol_surface"].append(vol_surfaces[i])
            all_AmericanP_NPVS_data["K"].append(eval_KTs[j][0])
            all_AmericanP_NPVS_data["T"].append(eval_KTs[j][1])
            all_AmericanP_NPVS_data["NPV"].append(npv)
            all_AmericanP_NPVS_data["S0"].append(S0_for_date)

    print(f"Processed {len(all_AmericanP_NPVS_data['quote_date'])} American put options.")
    print("error dates:", len(arb_date), arb_date)

    # 3. save data
    np.savez(
        f"{folder}/AmericanPut_pricing_data_{dataset_type}.npz",
        quote_dates=all_AmericanP_NPVS_data["quote_date"],
        vol_surfaces=all_AmericanP_NPVS_data["vol_surface"],
        K=all_AmericanP_NPVS_data["K"],
        T=all_AmericanP_NPVS_data["T"],
        NPV=all_AmericanP_NPVS_data["NPV"],
        UNDERLYING_LAST=all_AmericanP_NPVS_data["S0"],
    )
    print(f"American put data with {len(all_AmericanP_NPVS_data['NPV'])} samples saved to {folder}/AmericanPut_pricing_data_{dataset_type}.npz")
    return