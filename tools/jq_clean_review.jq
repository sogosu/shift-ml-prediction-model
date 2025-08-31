def srcstr: (.source // []) | join("");

def first_nonempty_line($s):
  ($s | split("\n") | map(select(length>0)) | (.[0] // ""));

def is_review: (.metadata // {}) | has("review");

def normalize_header_text($t):
  if ($t | test("(?i)library import")) then "Imports"
  elif ($t | test("(?i)cloud data access")) then "Data Loading (S3)"
  elif ($t | test("(?i)^[[:space:]]*review copy")) then "REVIEW COPY"
  elif ($t | test("(?i)data loading")) then "Data Loading"
  elif ($t | test("(?i)data combination|merge|concat")) then "Data Merging"
  elif ($t | test("(?i)feature engineering")) then "Feature Engineering"
  elif ($t | test("(?i)train/?test split|holdout|validation")) then "Train/Test Split"
  elif ($t | test("(?i)model training|training")) then "Model Training"
  elif ($t | test("(?i)model evaluation|evaluation|metrics")) then "Evaluation"
  elif ($t | test("(?i)visualization|plots|charts")) then "Visualization"
  elif ($t | test("(?i)empty cell")) then ""
  else ($t | gsub("\r"; "") | gsub("\t"; " ") | gsub("  +"; " ") | sub("^ "; ""))
  end;

def as_header_only($line):
  # Keep original heading level, replace text with normalized label
  ( $line | capture("^(?<prefix>[[:space:]]*#+)[[:space:]]*(?<text>.*)$")? ) as $cap |
  if $cap == null then [$line + "\n"]
  else
    (normalize_header_text($cap.text)) as $label |
    if $label == "" then [] else [($cap.prefix + " " + $label + "\n")] end
  end;

def is_claude_paragraph($s):
  # Heuristics for Claude-generated explanatory paragraphs (non-header)
  ($s | test("(?i)^[[:space:]]*This cell\\b") or
        test("(?i)^[[:space:]]*This notebook\\b") or
        test("(?i)scikit-learn.*backbone of the machine learning pipeline") or
        test("(?i)Pandas Library|NumPy Library|XGBoost Algorithm|Seaborn Visualization|Matplotlib Visualization") or
        test("(?i)CSV File Reading|Data Download|Categorical Encoding|Feature Scaling|Dimensionality Reduction")
  );

def process_cell(idx; c):
  if (c | is_review) or (c.metadata.review_copy // false) then c
  elif c.cell_type == "code" then c
  elif c.cell_type == "markdown" then
    (c|srcstr) as $s | (first_nonempty_line($s)) as $h1 |
    if ($h1 | test("^[[:space:]]*#")) then
      # Keep only a normalized header line; drop descriptive paragraphs
      (as_header_only($h1)) as $hdr |
      if ($hdr | length) == 0 then null else c + {source: $hdr} end
    elif is_claude_paragraph($s) then null
    else c
    end
  else c end;

.cells |= (to_entries | map(process_cell(.key; .value)) | map(select(. != null)))
