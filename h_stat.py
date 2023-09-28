from sklearn.inspection import partial_dependence
import itertools


def calc_h_stat(task_type, model, X, feats):
    if task_type == 'reg':
        h_stat = calc_h_stat_reg(model, X, feats)
    else:
        h_stat = calc_h_stat_cls (model, X, feats)

def compute_h_val(f_vals, selectedfeatures):
    numer_els = f_vals[tuple(selectedfeatures)].copy()
    denom_els = f_vals[tuple(selectedfeatures)].copy()
    sign = -1.0
    for n in range(len(selectedfeatures)-1, 0, -1):
        for subfeatures in itertools.combinations(selectedfeatures, n):
            numer_els += sign * f_vals[tuple(subfeatures)]
        sign *= -1.0
    numer = np.sum(numer_els**2)
    denom = np.sum(denom_els**2)
    return np.sqrt(numer/denom)

def calc_h_stat_reg(model, X, feats):
    def center(arr): 
        return arr - np.mean(arr)

    def cartesian_product(*arrays):
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[...,i] = a
        return arr.reshape(-1, la)

    def compute_f_vals_sklearn(model, X, feats=None, grid_resolution=10):

        def _pd_to_df(pde, feature_names):
            a = pde['values']
            df = pd.DataFrame(cartesian_product(*a))
            rename = {i: feature_names[i] for i in range(len(feature_names))}
            df.rename(columns=rename, inplace=True)
            df['preds'] = pde['average'].flatten()
            return df

        def _get_feat_idxs(feats):
            return [tuple(list(X.columns).index(f) for f in feats)]

        f_vals = {}
        if feats is None:
            feats = list(X.columns)

        # Calculate partial dependencies for full feature set
        pd_full = partial_dependence(
            model, X, _get_feat_idxs(feats), 
            grid_resolution=grid_resolution
        )

        # Establish the grid
        df_full = _pd_to_df(pd_full, feats)
        grid = df_full.drop('preds', axis=1)

        # Store
        f_vals[tuple(feats)] = center(df_full.preds.values)

        # Calculate partial dependencies for [1..SFL-1]
        for n in range(1, len(feats)):
            for subset in itertools.combinations(feats, n):
                pd_part = partial_dependence(
                    model, X, _get_feat_idxs(subset),
                    grid_resolution=grid_resolution
                )
                df_part = _pd_to_df(pd_part, subset)
                joined = pd.merge(grid, df_part, how='left')
                f_vals[tuple(subset)] = center(joined.preds.values)
        return f_vals

    f_val = compute_f_vals_sklearn(model, X, feats)
    h_val = compute_h_val(f_val, feats)
    return h_val

def calc_h_stat_cls(model, X, feats):
    def center(arr): 
        return arr - np.mean(arr)
    def compute_f_vals_manual(model, X, feats):
        def _partial_dependence(model, X, feats):
            P = X.copy()
            for f in P.columns:
                if f in feats: continue
                P.loc[:,f] = np.mean(P[f])
            # Assumes a regressor here, use return model.predict_proba(P)[:,1] for binary classification
            return model.predict_proba(P)[:,1]

        f_vals = {}
        if feats is None:
            feats = list(X.columns)

        # Calculate partial dependencies for full feature set
        full_preds = _partial_dependence(model, X, feats)
        f_vals[tuple(feats)] = center(full_preds)

        # Calculate partial dependencies for [1..SFL-1]
        for n in range(1, len(feats)):
            for subset in itertools.combinations(feats, n):
                pd_part = _partial_dependence(model, X, subset)
                f_vals[tuple(subset)] = center(pd_part)

        return f_vals
    
    f_vals = compute_f_vals_manual(model, X, feats)
    h_vals = compute_h_val(f_vals, feats)

    return h_vals
