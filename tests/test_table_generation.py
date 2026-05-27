from tsf_paperkit.reporting.table import write_table


def test_table_generation_markdown_and_latex(tmp_path):
    csv = tmp_path / "results.csv"
    csv.write_text("run_id,dataset,model,seq_len,pred_len,seed,mse,mae,rmse,mape,smape,train_time_sec,inference_time_sec,num_params\n1,toy,a,12,3,1,1,1,1,1,1,0,0,0\n2,toy,b,12,3,1,2,2,2,2,2,0,0,0\n")
    out = write_table(csv, tmp_path / "table.md", "markdown")
    text = out.read_text()
    assert "**1.0000**" in text
    assert "| model |" in text
    latex = write_table(csv, tmp_path / "table.tex", "latex").read_text()
    assert "\\begin{tabular}" in latex
    assert "\\textbf{1.0000}" in latex
