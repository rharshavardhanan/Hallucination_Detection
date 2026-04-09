from preprocess import load_data, build_features, split_and_resample
from train import train_lightgbm, train_catboost
from evaluate import get_metrics, print_results, compare_models
from visualize import (
    plot_class_distribution, plot_answer_length_distribution,
    plot_confusion_matrices, plot_roc_curves,
    plot_comparison_bar, plot_feature_importance
)
from predict import generate_submission


def main():
    print("Loading and preprocessing data...")
    train_df, test_df = load_data()

    print("Building features...")
    X, X_test, y, tfidf = build_features(train_df, test_df)

    print("Splitting and resampling...")
    X_train, X_val, y_train, y_val = split_and_resample(X, y)

    print(f"Train size: {X_train.shape[0]} | Val size: {X_val.shape[0]}")

    print("\nGenerating EDA charts...")
    plot_class_distribution(y)
    plot_answer_length_distribution(train_df)

    print("\nTraining LightGBM...")
    lgbm_model = train_lightgbm(X_train, y_train)

    print("Training CatBoost...")
    cat_model = train_catboost(X_train, y_train)

    print("\nEvaluating models...")
    lgbm_metrics = get_metrics(lgbm_model, X_val, y_val)
    cat_metrics = get_metrics(cat_model, X_val, y_val)

    print_results("LightGBM", lgbm_metrics)
    print_results("CatBoost", cat_metrics)
    compare_models(lgbm_metrics, cat_metrics)

    print("\nGenerating visualizations...")
    plot_confusion_matrices(y_val, lgbm_metrics['preds'], cat_metrics['preds'])
    plot_roc_curves(y_val, lgbm_metrics['proba'], cat_metrics['proba'])
    plot_comparison_bar(lgbm_metrics, cat_metrics)
    plot_feature_importance(lgbm_model, tfidf)

    print("\nGenerating submission file...")
    generate_submission(lgbm_model, cat_model, X_test, test_df)

    print("\nDone! All outputs saved to /outputs folder.")


if __name__ == "__main__":
    main()
