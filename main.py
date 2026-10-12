import os
from config.settings import cfg
from src.features import build_feature_stack, prepare_training_samples
from src.modeling import train_models
from src.evaluation import metrics
from src.mapping import generate_prob_maps

def main():
    ref, inside, stack = build_feature_stack(cfg, include_coping=False)
    data, X, y = prepare_training_samples(cfg, ref, inside, stack)
    fitted, X_tr, X_te, y_tr, y_te = train_models(cfg, X, y)

    rows = []
    for name, model in fitted.items():
        p_tr = model.predict_proba(X_tr)[:, 1]
        p_te = model.predict_proba(X_te)[:, 1]
        rows += [{"Model": name, "Dataset": "Training", **metrics(y_tr, p_tr)},
                 {"Model": name, "Dataset": "Testing", **metrics(y_te, p_te)}]

    prob_maps, class_breaks = generate_prob_maps(cfg, fitted, ref, inside, stack, list(stack.keys()))

    print("Completed Part A pipeline")
    return {
        "ref": ref,
        "inside": inside,
        "stack": stack,
        "X": X,
        "y": y,
        "fitted": fitted,
        "prob_maps": prob_maps,
        "class_breaks": class_breaks,
        "rows": rows,
    }

if __name__ == "__main__":
    main()
