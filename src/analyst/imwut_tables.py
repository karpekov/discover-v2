"""Paper tables for the analyst, written to results/analyst/imwut_tables/.

`analyst_review` is the review-tab summary: majority-vote quality of each step,
pooled and per home, with Fleiss' κ for each home's reviewer panel.
`analyst_funnel` is the question-level funnel (Milan, Aruba, Cairo, All).
`analyst_questions` is nine questions: rotated category, full wording, and a check
or cross for whether the analyst answered. `analyst_questions_all` is the same
for all 29, for the appendix.

  python src/analyst/imwut_tables.py
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analyst.config import PROJECT_ROOT, RESULTS_DIR  # noqa: E402
from analyst.review import _funnel, consensus_votes, fleiss_by_step  # noqa: E402

RUN = RESULTS_DIR / "runs" / "analyst_questions"
REVIEW = "review_v2"
OUT = RESULTS_DIR / "imwut_tables"
HOMES = ["milan", "aruba", "cairo"]
SCOPES = ["all", *HOMES]

# (kind, key, latex label). kind is "group", "rate", or "kappa".
# group label is (step title, raters' question). rate key is the step metric.
# kappa key is the step number (controls included).
ROWS = [
    ("group", None, (r"1\,$\cdot$\,Answerable?", r"Can this home's sensors answer the question?")),
    ("rate", "step1", r"\textbf{Agreed with \ToolName{}}"),
    ("rate", "step1_answerable", r"\hspace*{0.6em}on accepted questions"),
    ("rate", "step1_not_answerable", r"\hspace*{0.6em}on declined questions"),
    ("kappa", 1, r"\hspace*{0.6em}Fleiss' $\kappa$"),
    ("group", None, (r"2\,$\cdot$\,Queries", r"Is the retrieval query relevant?")),
    ("rate", "step2", r"\textbf{Relevant queries}"),
    ("rate", "step2_decoy", r"\hspace*{0.6em}Decoy queries$^{\downarrow}$"),
    ("kappa", 2, r"\hspace*{0.6em}Fleiss' $\kappa$"),
    ("group", None, (r"3\,$\cdot$\,Retrieval", r"Does the retrieved window show the behavior?")),
    ("rate", "step3", r"\textbf{Relevant windows}"),
    ("rate", "step3_top", r"\hspace*{0.6em}$\hookrightarrow$\,Top-ranked windows"),
    ("rate", "step3_random", r"\hangindent=1.9em\hspace*{0.6em}$\hookrightarrow$\,Random windows within threshold"),
    ("rate", "step3_negative", r"\hspace*{0.6em}Low-similarity windows$^{\downarrow}$"),
    ("kappa", 3, r"\hspace*{0.6em}Fleiss' $\kappa$"),
    ("group", None, (r"4\,$\cdot$\,Findings", r"Is the finding supported?")),
    ("rate", "step4", r"\textbf{Supported findings}"),
    ("kappa", 4, r"\hspace*{0.6em}Fleiss' $\kappa$"),
]

# {scope.metric} placeholders are filled from the table's own rates (rounded %), so the text tracks the data.
NOTE = (
    r"Three independent raters judged the output of each step of \ToolName{} "
    r"(majority vote; yes\,/\,decided per cell). The results are strong where the pipeline acts on a "
    r"question: {all.step2}\% of retrieval queries were judged relevant, against {all.step2_decoy}\% of "
    r"decoy queries mixed among them, and {all.step3_top}\% of top-ranked retrieved windows showed the "
    r"intended behavior, against {all.step3_negative}\% of low-similarity windows. \ToolName{} is weaker "
    r"at deciding what a home can answer: raters agreed with {all.step1_answerable}\% of the questions it "
    r"accepted but only {all.step1_not_answerable}\% of those it declined "
    r"({cairo.step1_not_answerable}\% in Cairo, which has motion sensors only), and precision drops from "
    r"top-ranked to randomly drawn matched windows ({all.step3_top}\% vs.\ {all.step3_random}\%). "
    r"Step~3 rates four top-ranked, two random above-threshold, and two low-similarity windows per "
    r"concept; rows marked $^{\downarrow}$ are controls and should be low. Fleiss' $\kappa$ is computed "
    r"over items decided by all three raters, including controls."
)


def _fill_note(note: str, by_scope: dict) -> str:
    def pct(m: re.Match) -> str:
        c = by_scope[m.group(1)][m.group(2)]
        return f"{int(100 * c['yes'] / (c['yes'] + c['no']) + 0.5)}"
    return re.sub(r"\{(\w+)\.(\w+)\}", pct, note)


def _as_bool(v) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes"}


def _load() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    key = pd.concat([pd.read_csv(p) for p in sorted((RUN / "review" / f"{REVIEW}_key").glob("*_key.csv"))],
                    ignore_index=True)
    ratings = pd.concat([pd.read_csv(p) for p in sorted((RUN / "review" / REVIEW).glob("*.csv"))],
                        ignore_index=True)
    ratings["rating"] = ratings["rating"].astype(str).str.lower().str.strip()
    all_ratings = ratings.drop_duplicates(["reviewer", "item_id"], keep="last")
    key["analyst_answerable"] = key["analyst_answerable"].map(_as_bool)
    key["stratum"] = key["stratum"].fillna("").astype(str)
    consensus = consensus_votes(all_ratings)
    return key, consensus, all_ratings


def _counts(sub: pd.DataFrame, consensus: dict, *, agree: bool) -> dict:
    yes = no = unsure = 0
    for row in sub.itertuples(index=False):
        vote = consensus.get(row.item_id)
        if vote is None:
            continue
        if agree and vote in ("yes", "no"):
            vote = "yes" if (vote == "yes") == bool(row.analyst_answerable) else "no"
        if vote == "yes":
            yes += 1
        elif vote == "no":
            no += 1
        else:
            unsure += 1
    return {"yes": yes, "no": no, "unsure": unsure}


def _metrics(key: pd.DataFrame, consensus: dict) -> dict:
    s2 = key["stratum"] != "decoy"
    groups = {
        "step1": (key[key.step == 1], True),
        "step1_answerable": (key[(key.step == 1) & key.analyst_answerable], True),
        "step1_not_answerable": (key[(key.step == 1) & ~key.analyst_answerable], True),
        "step2": (key[(key.step == 2) & s2], False),
        "step2_decoy": (key[(key.step == 2) & (key.stratum == "decoy")], False),
        "step3": (key[(key.step == 3) & key.stratum.isin(["top", "random"])], False),
        "step3_top": (key[(key.step == 3) & (key.stratum == "top")], False),
        "step3_random": (key[(key.step == 3) & (key.stratum == "random")], False),
        "step3_negative": (key[(key.step == 3) & (key.stratum == "negative")], False),
        "step4": (key[key.step == 4], False),
    }
    return {name: _counts(sub, consensus, agree=agree) for name, (sub, agree) in groups.items()}


def _rate_cell(c: dict, bold: bool = False) -> str:
    n = c["yes"] + c["no"]
    if not n:
        return r"\textemdash"
    pct = rf"{100.0 * c['yes'] / n:.1f}\%"
    if bold:
        pct = rf"\textbf{{{pct}}}"
    # The count is \tiny and the gap narrow so the widest (bold) cell stays on one line inside 14%.
    return rf"\mbox{{{pct}\hspace{{0.15em}}{{\tiny({c['yes']}/{n})}}}}"


def _kappa_cell(k: dict | None) -> str:
    if not k or not k["n"]:
        return r"\textemdash"
    return f"{k['kappa']:.2f}"


def build_tabular(by_scope: dict, kappa_by_scope: dict) -> list[str]:
    headers = " & ".join([r"\textbf{Pipeline Step}"] + [rf"\textbf{{{s.capitalize()}}}" for s in SCOPES])
    # Step is 44% of the line; All, Milan, Aruba, and Cairo are 14% each.
    # All is a p column so \columncolor fills the whole column. \tabcolsep 0 keeps the five widths on \linewidth.
    lines = [
        r"\setlength{\tabcolsep}{0pt}",
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.44\linewidth}"
        r">{\columncolor{black!5}\centering\arraybackslash}p{0.14\linewidth}"
        r"*{3}{w{c}{0.14\linewidth}}@{}}",
        r"\toprule",
        headers + r" \\",
        r"\midrule",
    ]
    first = False
    for kind, key, label in ROWS:
        if kind == "group":
            first = True
            # Title and the raters' question on one row, in column 1.
            # A faint \hline (no booktabs padding) separates steps without breaking the All shading.
            title, question = label
            if lines[-1] != r"\midrule":
                lines.append(r"\arrayrulecolor{black!15}\hline\arrayrulecolor{black}")
            lines.append(r"\textbf{" + title + r"}\enspace{\footnotesize\textit{" + question + r"}} & & & & \\")
            continue
        if kind == "rate":
            # The first rate of each step is its headline number.
            cells = [_rate_cell(by_scope[s][key], bold=first) for s in SCOPES]
            first = False
            end = r" \\"
        else:
            cells = [_kappa_cell((kappa_by_scope.get(s) or {}).get(f"step{key}")) for s in SCOPES]
            end = r" \\"
        lines.append(" & ".join([label, *cells]) + end)
    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines


def _tex_fragment(tabular: list[str], note: str, *, label: str, env: str = "table*") -> str:
    return "\n".join([
        rf"\begin{{{env}}}[t]",
        r"\centering",
        r"\caption{" + note + "}",
        rf"\label{{{label}}}",
        r"\small",
        r"\renewcommand{\arraystretch}{1.05}",
        *tabular,
        rf"\end{{{env}}}",
        "",
    ])


def _standalone(tabular: list[str], note: str, *, width_cm: float = 16.2, page_width: str | None = None) -> str:
    width = page_width or f"{width_cm}cm"
    note_width = rf"\dimexpr{width}-6pt\relax" if page_width else f"{width_cm - 0.2}cm"
    return "\n".join([
        r"\documentclass[border=8pt]{standalone}",
        r"\usepackage{booktabs}",
        r"\providecommand{\ToolName}{\textbf{SensorQ}}",
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
        r"\usepackage{array}",
        r"\usepackage[table]{xcolor}",
        r"\usepackage{graphicx}",
        r"\usepackage{multirow}",
        r"\begin{document}",
        rf"\begin{{minipage}}{{{width}}}",
        r"\centering",
        r"\small",
        r"\renewcommand{\arraystretch}{1.05}",
        *tabular,
        r"\vspace{6pt}",
        rf"\parbox{{{note_width}}}{{\footnotesize " + note + "}",
        r"\end{minipage}",
        r"\end{document}",
        "",
    ])


def _compile(standalone: str, pdf_path: Path, png_path: Path) -> None:
    pdflatex = shutil.which("pdflatex") or next(
        (p for p in (
            "/Library/TeX/texbin/pdflatex",
            "/usr/local/texlive/2026/bin/universal-darwin/pdflatex",
            "/usr/local/texlive/2025/bin/universal-darwin/pdflatex",
            "/usr/local/texlive/2024/bin/universal-darwin/pdflatex",
        ) if Path(p).exists()),
        None,
    )
    if not pdflatex:
        print("[WARN] pdflatex not found; wrote the .tex only", file=sys.stderr)
        return
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tex = tmp / "table.tex"
        tex.write_text(standalone, encoding="utf-8")
        result = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", str(tex)],
            cwd=tmp, capture_output=True, text=True,
        )
        pdf = tmp / "table.pdf"
        if not pdf.exists():
            tail = (result.stdout or result.stderr or "")[-1200:]
            print(f"[WARN] pdflatex failed:\n{tail}", file=sys.stderr)
            return
        shutil.copy(pdf, pdf_path)
        print(f"PDF  {pdf_path}")
        pdftoppm = shutil.which("pdftoppm")
        if not pdftoppm:
            print("[WARN] pdftoppm not found; PDF written, PNG skipped", file=sys.stderr)
            return
        subprocess.run(
            [pdftoppm, "-r", "300", "-png", "-singlefile", str(pdf), str(tmp / "out")],
            check=True,
        )
        shutil.copy(tmp / "out.png", png_path)
        print(f"PNG  {png_path}")


def write_review_table() -> Path:
    key, consensus, ratings = _load()
    by_scope = {"all": _metrics(key, consensus)}
    for home in HOMES:
        by_scope[home] = _metrics(key[key.home == home], consensus)
    kappa_by_scope = {"all": fleiss_by_step(key, ratings)}
    for home in HOMES:
        kappa_by_scope[home] = fleiss_by_step(key[key.home == home], ratings)
    tabular = build_tabular(by_scope, kappa_by_scope)
    note = _fill_note(NOTE, by_scope)
    OUT.mkdir(parents=True, exist_ok=True)
    kappa_path = OUT / "analyst_review_kappa.json"
    kappa_path.write_text(json.dumps(kappa_by_scope, indent=1), encoding="utf-8")
    print(f"JSON {kappa_path}")
    OUT.mkdir(parents=True, exist_ok=True)
    tex_path = OUT / "analyst_review.tex"
    tex_path.write_text(
        "% requires array, xcolor (table option; load xcolor before arydshln)\n"
        + _tex_fragment(tabular, note, label="tab:analyst_review"),
        encoding="utf-8",
    )
    print(f"TEX  {tex_path}")
    _compile(
        _standalone(tabular, note, page_width="394pt"),
        OUT / "analyst_review.pdf",
        OUT / "analyst_review.png",
    )
    return tex_path


# Question-level funnel. The first five rows partition the 29 questions per home.
# The last three are how many of the agreed-answerable questions also passed later steps.
FUNNEL_ROWS = [
    ("s1_no_ok", "Not answerable, agreed"),
    ("s1_missed", "Missed"),
    ("s1_false_ans", "Wrongly judged answerable"),
    ("s1_unsure", "Step-1 unsure"),
    ("s1_ok", "Agreed answerable"),
    ("s2_ok", r"\ldots\ queries passed"),
    ("s3_ok", r"\ldots\ retrieval passed"),
    ("s4_ok", r"\ldots\ findings passed"),
]
FUNNEL_SCOPES = [*HOMES, "all"]
FUNNEL_NOTE = (
    r"Question-level review funnel, majority vote. "
    r"Each home has 29 questions (87 in all). "
    r"The first five rows partition those questions. "
    r"Missed means the vote said answerable and \ToolName{} said not; "
    r"wrongly judged answerable is the reverse. "
    r"Step-1 unsure is a tied vote among the home's reviewers, and those questions stop. "
    r"The last three rows count how many of the agreed-answerable questions also passed "
    r"the query check, then retrieval, then the findings."
)


def _funnel_nodes(key: pd.DataFrame, consensus: dict) -> dict:
    return _funnel(key, consensus)["nodes"]


def build_funnel_tabular(by_scope: dict) -> list[str]:
    headers = " & ".join(
        [r"\textbf{Outcome}"] + [rf"\textbf{{{s.capitalize()}}}" for s in FUNNEL_SCOPES]
    )
    lines = [
        r"\setlength{\tabcolsep}{8pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        headers + r" \\",
        r"\midrule",
    ]
    for key, label in FUNNEL_ROWS:
        if key == "s1_ok":
            lines.append(r"\midrule")
        cells = [str(int(by_scope[s].get(key, 0))) for s in FUNNEL_SCOPES]
        lines.append(" & ".join([label, *cells]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines


def write_funnel_table(key: pd.DataFrame | None = None, consensus: dict | None = None) -> Path:
    if key is None or consensus is None:
        key, consensus, _ratings = _load()
    by_scope = {"all": _funnel_nodes(key, consensus)}
    for home in HOMES:
        by_scope[home] = _funnel_nodes(key[key.home == home], consensus)
    tabular = build_funnel_tabular(by_scope)
    OUT.mkdir(parents=True, exist_ok=True)
    tex_path = OUT / "analyst_funnel.tex"
    tex_path.write_text(
        _tex_fragment(tabular, FUNNEL_NOTE, label="tab:analyst_funnel", env="table"),
        encoding="utf-8",
    )
    print(f"TEX  {tex_path}")
    _compile(
        _standalone(tabular, FUNNEL_NOTE, width_cm=12.4),
        OUT / "analyst_funnel.pdf",
        OUT / "analyst_funnel.png",
    )
    return tex_path


# Main-text selection, by question number. The appendix table lists every question.
QUESTION_SELECTION = [4, 15, 9, 11, 21, 20, 26, 27, 28]
QUESTIONS_NOTE = (
    r"Nine of the 29 caregiver questions. For each question and home, "
    r"\ToolName{} either analyzes the question ($\checkmark$) or declines it "
    r"({\color{black!30}$\times$}) because the home's sensors cannot observe the "
    r"behavior, and explains why. The same question can be analyzable in one home "
    r"and not another: only Cairo has sensors on its stairs. The last row counts "
    r"all 29 questions. The full list is in Table~\ref{tab:analyst_questions_all}; "
    r"\cref{sec:results_analyst} evaluates both decisions."
)
QUESTIONS_ALL_NOTE = (
    r"All 29 caregiver questions. "
    r"A check means \ToolName{} treated the question as answerable in that home. "
    r"The last row counts how many of the 29 questions \ToolName{} answered in each home. "
    r"$\checkmark$ marks questions \ToolName{} analyzed in that home and "
    r"{\color{black!30}$\times$} questions it declined because the home's sensors "
    r"cannot observe the behavior."
)
# Short reasons shown after a declined question in the main table.
QUESTION_REASONS = {"hydration": r"{\scriptsize\color{black!55}(no sensor observes drinking)}"}
MARK_MACROS = [r"\newcommand{\yes}{$\checkmark$}", r"\newcommand{\no}{{\color{black!30}$\times$}}"]


def _tex_escape(text: str) -> str:
    return (text.replace("\\", r"\textbackslash{}")
            .replace("&", r"\&").replace("%", r"\%")
            .replace("#", r"\#").replace("_", r"\_"))


def _load_questions() -> list[dict]:
    qfile = json.loads((PROJECT_ROOT / "metadata" / "analyst_questions.json").read_text())
    return sorted(qfile["questions"], key=lambda q: q["num"])


def _analyst_answered() -> dict:
    """(home, question_id) -> analyzed. Declined = every concept in plan.json has detectable=false."""
    answered = {}
    for plan in sorted(RUN.glob("*/*/plan.json")):
        home, qid = plan.parent.parent.name, plan.parent.name
        if home in HOMES:
            concepts = json.loads(plan.read_text()).get("concepts") or []
            answered[(home, qid)] = any(c.get("detectable") for c in concepts)
    key = pd.concat(
        [pd.read_csv(p) for p in sorted((RUN / "review" / f"{REVIEW}_key").glob("*_key.csv"))],
        ignore_index=True,
    )
    step1 = key[key.step == 1]
    diff = [(r.home, r.question_id) for r in step1.itertuples(index=False)
            if answered.get((r.home, r.question_id)) != _as_bool(r.analyst_answerable)]
    if diff:
        raise ValueError(f"plan.json and the review key disagree on answerable: {diff[:6]}")
    return answered


def _mark(answered: bool) -> str:
    return r"\yes" if answered else r"\no"


def _answered_counts(questions: list[dict], answered: dict) -> dict:
    """home -> (analyzed, declined) over all questions; the two must add up to the question count."""
    counts = {}
    for home in HOMES:
        keys = [f"{q['num']:02d}_{q['id']}" for q in questions]
        yes = sum(1 for key in keys if answered[(home, key)])
        counts[home] = (yes, len(keys) - yes)
        assert sum(counts[home]) == len(questions) == 29, (home, counts[home])
    return counts


def _count_cell(yes: int, no: int) -> str:
    return rf"\begin{{tabular}}[t]{{@{{}}c@{{}}}}{yes} \yes\\{no} \no\end{{tabular}}"


def build_questions_tabular(
    rows: list[dict],
    answered: dict,
    *,
    summary_from: list[dict],
    summary_label: str,
    ellipsis: bool = False,
    reasons: dict | None = None,
) -> list[str]:
    grouped: list[tuple[str, list[dict]]] = []
    for q in rows:
        if not grouped or grouped[-1][0] != q["category"]:
            grouped.append((q["category"], []))
        grouped[-1][1].append(q)
    headers = " & ".join(
        ["", r"\textbf{Caregiver's Question}"] + [rf"\textbf{{{h.capitalize()}}}" for h in HOMES]
    )
    lines = [
        *MARK_MACROS,
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        # Question column is whatever the page has left, so the table stays within \linewidth.
        r"\begin{tabular}{@{}c >{\raggedright\arraybackslash}p{\dimexpr\linewidth-7.6cm\relax} ccc@{}}",
        r"\toprule",
        headers + r" \\",
        r"\midrule",
    ]
    for cat, group in grouped:
        span = len(group) + (1 if ellipsis else 0)
        for i, q in enumerate(group):
            key = f"{q['num']:02d}_{q['id']}"
            cat_cell = rf"\multirow{{{span}}}{{*}}{{\rotatebox{{90}}{{{cat}}}}}" if i == 0 else ""
            cells = [_mark(answered[(home, key)]) for home in HOMES]
            text = _tex_escape(q["question"])
            if reasons and q["id"] in reasons:
                text += " " + reasons[q["id"]]
            lines.append(" & ".join([cat_cell, text, *cells]) + r" \\")
        if ellipsis:
            lines.append(r" & \ldots & & & \\")
        if cat != grouped[-1][0]:
            lines.append(r"\midrule")
    lines.append(r"\midrule")
    counts = _answered_counts(summary_from, answered)
    cells = [_count_cell(*counts[home]) for home in HOMES]
    lines.append(" & ".join(["", rf"\textbf{{{summary_label}}}", *cells]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines


def _write_question_table(
    rows: list[dict],
    answered: dict,
    *,
    stem: str,
    label: str,
    note: str,
    standalone_note: str,
    width_cm: float,
    summary_from: list[dict],
    summary_label: str,
    ellipsis: bool = False,
    reasons: dict | None = None,
    body_size: str = "",
) -> Path:
    missing = [(home, f"{q['num']:02d}_{q['id']}") for q in summary_from for home in HOMES
               if (home, f"{q['num']:02d}_{q['id']}") not in answered]
    if missing:
        raise KeyError(f"missing step-1 row for {missing[:6]}")
    tabular = build_questions_tabular(
        rows, answered, summary_from=summary_from, summary_label=summary_label,
        ellipsis=ellipsis, reasons=reasons,
    )
    if body_size:
        tabular.insert(0, body_size)
    OUT.mkdir(parents=True, exist_ok=True)
    tex_path = OUT / f"{stem}.tex"
    fragment = "\n".join([
        r"% requires booktabs, amssymb, array, graphicx, multirow, xcolor",
        _tex_fragment(tabular, note, label=label, env="table*"),
    ])
    tex_path.write_text(fragment, encoding="utf-8")
    print(f"TEX  {tex_path}")
    _compile(
        _standalone(tabular, standalone_note, width_cm=width_cm),
        OUT / f"{stem}.pdf",
        OUT / f"{stem}.png",
    )
    return tex_path


def write_questions_table(questions: list[dict], answered: dict) -> Path:
    by_num = {q["num"]: q for q in questions}
    rows = [by_num[n] for n in QUESTION_SELECTION]
    standalone_note = (QUESTIONS_NOTE.replace(r"Table~\ref{tab:analyst_questions_all}", "the appendix")
                       .replace(r"\cref{sec:results_analyst}", "the results section"))
    return _write_question_table(
        rows, answered,
        stem="analyst_questions",
        label="tab:analyst_questions",
        note=QUESTIONS_NOTE,
        standalone_note=standalone_note,
        width_cm=16.5,
        summary_from=questions,
        summary_label="All 29 questions",
        ellipsis=True,
        reasons=QUESTION_REASONS,
    )


def write_questions_all_table(questions: list[dict], answered: dict) -> Path:
    return _write_question_table(
        questions, answered,
        stem="analyst_questions_all",
        label="tab:analyst_questions_all",
        note=QUESTIONS_ALL_NOTE,
        standalone_note=QUESTIONS_ALL_NOTE,
        width_cm=16.5,
        summary_from=questions,
        summary_label="Total",
        body_size=r"\footnotesize",
    )


if __name__ == "__main__":
    key, consensus, _ratings = _load()
    write_review_table()
    write_funnel_table(key, consensus)
    questions = _load_questions()
    answered = _analyst_answered()
    main_tex = write_questions_table(questions, answered)
    all_tex = write_questions_all_table(questions, answered)
    cell = r"\{@\{\}c@\{\}\}(\d+) \\yes\\\\(\d+) \\no"
    main_counts = re.findall(cell, main_tex.read_text())
    all_counts = re.findall(cell, all_tex.read_text())
    assert len(main_counts) == len(HOMES) and main_counts == all_counts, (main_counts, all_counts)
