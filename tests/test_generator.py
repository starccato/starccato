from staccato.nn import generate_signals
import os


def test_cli_generate(cli_runner):
    result = cli_runner.invoke(generate_signals, ["--n", "10"])
    assert result.exit_code == 0
    assert "Generated 10 signals and saved to signals.txt" in result.output


def test_python_generate(tmpdir):
    generate_signals(n=10, filename=f"{tmpdir}/generated_signals.txt")
    assert os.path.exists(f"{tmpdir}/generated_signals.txt")

