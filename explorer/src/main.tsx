import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { initialize, predict } from "./inference";
import type { Feature, Release } from "./types";
import "./style.css";

const source = "https://github.com/xSanboyzx/MindMetrics-ML";
const format = (value: number, decimals = 2) => value.toFixed(decimals);

function Mark() {
  return (
    <svg viewBox="0 0 40 40" fill="none" aria-hidden="true">
      <rect width="40" height="40" rx="11" fill="currentColor" />
      <path
        d="M9 26V15l8 8 8-13v16m7-11v11"
        stroke="#e6f3bc"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function Scatter({ data }: { data: Release["scatter"] }) {
  const x = (value: number) => 45 + (value / 10) * 395;
  const y = (value: number) => 246 - (value / 10) * 218;
  return (
    <svg
      className="scatter"
      viewBox="0 0 470 292"
      role="img"
      aria-label="525 held-out synthetic records: actual happiness score versus neural network prediction. The diagonal indicates perfect agreement."
    >
      {[0, 2, 4, 6, 8, 10].map((tick) => (
        <g key={tick}>
          <path
            d={`M45 ${y(tick)}H440 M${x(tick)} 28V246`}
            className="gridline"
          />
          <text x="33" y={y(tick) + 4} textAnchor="end">
            {tick}
          </text>
          <text x={x(tick)} y="265" textAnchor="middle">
            {tick}
          </text>
        </g>
      ))}
      <path d="M45 246L440 28" className="diagonal" />
      {data.map((point, i) => (
        <circle
          key={i}
          cx={x(point.actual)}
          cy={y(point.predicted)}
          r="2.5"
          className="dot"
        >
          <title>
            Actual {format(point.actual)}, predicted {format(point.predicted)}
          </title>
        </circle>
      ))}
      <text x="245" y="287" textAnchor="middle">
        Actual dataset score
      </text>
      <text transform="translate(12 140) rotate(-90)" textAnchor="middle">
        Model prediction
      </text>
    </svg>
  );
}

function Explorer({ release }: { release: Release }) {
  const [presetIndex, setPresetIndex] = useState(0);
  const preset = release.presets[presetIndex];
  const [values, setValues] = useState(preset.values);
  const [compared, setCompared] = useState(preset.values);
  const [scores, setScores] = useState<number[] | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">(
    "loading",
  );
  const [busy, setBusy] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const [resetVersion, setResetVersion] = useState(0);
  const [invalid, setInvalid] = useState<Record<string, boolean>>({});
  useEffect(() => {
    let cancelled = false;
    setStatus("loading");
    setScores(null);
    initialize()
      .then(() => predict([preset.values, preset.values]))
      .then((result) => {
        if (!cancelled) {
          setScores(result);
          setStatus("ready");
        }
      })
      .catch(() => {
        if (!cancelled) setStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, [preset, attempt]);
  const select = (index: number) => {
    setPresetIndex(index);
    setValues(release.presets[index].values);
    setCompared(release.presets[index].values);
    setInvalid({});
  };
  const reset = () => {
    setResetVersion((v) => v + 1);
    setValues(preset.values);
    setCompared(preset.values);
    setInvalid({});
    if (scores) setScores([scores[0], scores[0]]);
  };
  const change = (index: number, value: number) =>
    setValues((current) => current.map((v, i) => (i === index ? value : v)));
  const compare = async () => {
    setBusy(true);
    try {
      setScores(await predict([preset.values, values]));
      setCompared([...values]);
    } catch {
      setStatus("error");
    } finally {
      setBusy(false);
    }
  };
  const changes = compared.flatMap((value, i) =>
    value === preset.values[i]
      ? []
      : [{ feature: release.features[i], value, before: preset.values[i] }],
  );
  const pending = values.some((value, i) => value !== compared[i]);
  const groups = [...new Set(release.features.map((f) => f.group))];
  const hasInvalid = Object.values(invalid).some(Boolean);
  const outOfRange = scores?.some((score) => score < 0 || score > 10);

  return (
    <>
      <section
        id="explore"
        className="explore-section"
        aria-labelledby="explore-title"
      >
        <div className="section-heading">
          <div>
            <p className="eyebrow">01 / THE EXPLORER</p>
            <h2 id="explore-title">A small change. A different prediction.</h2>
          </div>
          <p>
            Start with a fictional profile. <br />
            Adjust its inputs. See what the model does.
          </p>
        </div>
        <div
          className="presets"
          role="group"
          aria-label="Fictional example profiles"
        >
          {release.presets.map((item, index) => (
            <button
              type="button"
              key={item.id}
              className={`preset ${index === presetIndex ? "selected" : ""}`}
              aria-pressed={index === presetIndex}
              onClick={() => select(index)}
              disabled={busy}
            >
              <span className="preset-number">0{index + 1}</span>
              <span>
                <strong>{item.name}</strong>
                <small>{item.description}</small>
              </span>
              <span className="selection-dot" aria-hidden="true" />
            </button>
          ))}
        </div>
        <div className="workbench">
          <form
            className="controls"
            onSubmit={(event) => {
              event.preventDefault();
              void compare();
            }}
          >
            <div className="control-heading">
              <div>
                <h3>Shape the scenario</h3>
                <p>15 inputs · ranges observed in the dataset</p>
              </div>
              <button
                type="button"
                className="text-button"
                onClick={reset}
                disabled={busy}
              >
                Reset changes ↺
              </button>
            </div>
            {groups.map((group, groupIndex) => (
              <fieldset key={group} disabled={busy}>
                <legend>
                  <span className="group-number">0{groupIndex + 1}</span>
                  {group}
                </legend>
                <div className="field-grid">
                  {release.features.map((feature, index) =>
                    feature.group !== group ? null : (
                      <FeatureControl
                        key={`${preset.id}-${resetVersion}-${feature.key}`}
                        feature={feature}
                        value={values[index]}
                        onChange={(value) => change(index, value)}
                        onValidity={(valid) =>
                          setInvalid((current) => ({
                            ...current,
                            [feature.key]: !valid,
                          }))
                        }
                      />
                    ),
                  )}
                </div>
              </fieldset>
            ))}
            <div className="compare-bar">
              <p>Inputs stay in this browser session.</p>
              <button
                className="primary-button"
                disabled={status !== "ready" || busy || hasInvalid}
                type="submit"
              >
                {busy ? "Comparing…" : "Compare predictions"}{" "}
                <span aria-hidden="true">↗</span>
              </button>
            </div>
          </form>
          <aside className="result" aria-labelledby="result-title">
            <div className="result-top">
              <span className="eyebrow">MODEL OUTPUT</span>
              <span className="live-badge">
                <i />
                ON-DEVICE
              </span>
            </div>
            <h3 id="result-title">The model’s view.</h3>
            <p className="result-description">
              Estimated synthetic happiness score
            </p>
            <div aria-live="polite" aria-atomic="true">
              {status === "loading" && (
                <p className="status-message">
                  Loading the model into your browser…
                </p>
              )}
              {status === "error" && (
                <div role="alert">
                  <p>
                    The model couldn’t load or run. Check your connection and
                    try again.
                  </p>
                  <button
                    className="light-button"
                    onClick={() => setAttempt((a) => a + 1)}
                  >
                    Retry model
                  </button>
                </div>
              )}
              {status === "ready" && scores && (
                <>
                  <div className="score" data-testid="adjusted-score">
                    {format(scores[1])}
                    <span>/ 10</span>
                  </div>
                  <div className="score-track" role="presentation">
                    <span
                      style={{
                        width: `${Math.max(0, Math.min(100, scores[1] * 10))}%`,
                      }}
                    />
                  </div>
                  <div className="comparison">
                    <div>
                      <small>Fictional baseline</small>
                      <strong data-testid="baseline-score">
                        {format(scores[0])}
                      </strong>
                    </div>
                    <div>
                      <small>Prediction difference</small>
                      <strong data-testid="score-delta">
                        {scores[1] - scores[0] >= 0 ? "+" : ""}
                        {format(scores[1] - scores[0])}
                      </strong>
                    </div>
                  </div>
                  {outOfRange && (
                    <p className="result-note">
                      Output outside the dataset’s 0–10 target range. The number
                      is shown without clipping.
                    </p>
                  )}
                  {pending && (
                    <p className="pending">
                      Inputs changed. Compare to update this result.
                    </p>
                  )}
                </>
              )}
            </div>
            <div className="changes">
              <h4>What changed</h4>
              {changes.length ? (
                <ul>
                  {changes.map(({ feature, value, before }) => (
                    <li key={feature.key}>
                      <span>{feature.label}</span>
                      <strong>
                        {format(before, feature.decimals)} →{" "}
                        {format(value, feature.decimals)}
                      </strong>
                    </li>
                  ))}
                </ul>
              ) : (
                <p>
                  Your selected profile is the baseline. Move a control to
                  explore a different scenario.
                </p>
              )}
            </div>
            <div className="result-note">
              <span aria-hidden="true">↳</span>
              <p>
                This shows an association learned from synthetic data. Changing
                an input here does not establish what would happen in real life.
              </p>
            </div>
          </aside>
        </div>
        <p className="under-note">
          Every profile is fictional. Valid input ranges do not guarantee that a
          combination resembles the training data. Score variables are dataset
          indicators, not clinical instruments.
        </p>
      </section>
      <section
        id="evaluation"
        className="evaluation-section"
        aria-labelledby="evaluation-title"
      >
        <div className="section-heading">
          <div>
            <p className="eyebrow">02 / SHOW YOUR WORK</p>
            <h2 id="evaluation-title">How well does it actually work?</h2>
          </div>
          <p>
            Measured on 525 held-out synthetic records. <br />
            Same test split for every model.
          </p>
        </div>
        <div className="evaluation-grid">
          <div className="evaluation-copy">
            <h3>A benchmark, with context.</h3>
            <p>
              The neural network learns useful patterns in this dataset. A
              simple linear model gives us a reality check: added complexity has
              to earn its place.
            </p>
            <div className="table-wrap">
              <table>
                <caption>
                  Test-set performance · lower MAE / RMSE is better; higher R²
                  is better
                </caption>
                <thead>
                  <tr>
                    <th scope="col">Model</th>
                    <th scope="col">MAE</th>
                    <th scope="col">RMSE</th>
                    <th scope="col">R²</th>
                  </tr>
                </thead>
                <tbody>
                  {release.metrics.map((metric) => (
                    <tr
                      key={metric.name}
                      className={
                        metric.name.startsWith("Residual") ? "network-row" : ""
                      }
                    >
                      <th scope="row">{metric.name}</th>
                      <td>{format(metric.mae, 3)}</td>
                      <td>{format(metric.rmse, 3)}</td>
                      <td>{format(metric.r2, 3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="metric-explainer">
              MAE is the average absolute error in score points. RMSE gives
              larger errors more weight. R² compares explained variation against
              a constant prediction.
            </p>
          </div>
          <figure className="scatter-card">
            <div className="chart-title">
              <h3>Predicted vs. actual</h3>
              <span>RESIDUAL NETWORK</span>
            </div>
            <Scatter data={release.scatter} />
            <figcaption>
              Each dot is one synthetic test record. Closer to the diagonal
              means a smaller prediction error.
            </figcaption>
          </figure>
        </div>
        <div className="method-strip">
          <div>
            <strong>3,500</strong>
            <span>synthetic records</span>
          </div>
          <div>
            <strong>15</strong>
            <span>ordered inputs</span>
          </div>
          <div>
            <strong>70 / 15 / 15</strong>
            <span>train / validation / test</span>
          </div>
          <div>
            <strong>Training only</strong>
            <span>scaler fitted on 2,450 records</span>
          </div>
        </div>
      </section>
      <section
        id="behind"
        className="behind-section"
        aria-labelledby="behind-title"
      >
        <div>
          <p className="eyebrow">03 / UNDER THE HOOD</p>
          <h2 id="behind-title">
            A real model.
            <br />
            An open notebook.
          </h2>
          <p>
            Originally a two-person machine learning course project. This
            release connects the trained PyTorch network to an inspectable
            browser experience.
          </p>
          <a
            className="source-link"
            href={`${source}/tree/showcase/model-explorer`}
            target="_blank"
            rel="noopener noreferrer"
          >
            Explore the source ↗
          </a>
        </div>
        <div className="pipeline">
          {[
            [
              "01",
              "Prepare",
              "15 ordered features. Training-only standardization.",
            ],
            ["02", "Learn", "Residual blocks, batch normalization, dropout."],
            [
              "03",
              "Verify",
              "Held-out evaluation, simple baselines, parity checks.",
            ],
            [
              "04",
              "Explore",
              "ONNX + WebAssembly. Predictions on your device.",
            ],
          ].map(([number, title, copy]) => (
            <div className="pipeline-step" key={number}>
              <span>{number}</span>
              <div>
                <h3>{title}</h3>
                <p>{copy}</p>
              </div>
            </div>
          ))}
        </div>
      </section>
      <section className="limitations">
        <h3>What this demo can tell you</h3>
        <p>
          It demonstrates a reproducible inference pipeline and patterns in a
          synthetic benchmark. It has no validation on real people, no causal
          interpretation, and no clinical use. The original notebook’s
          checkpoint-selection bug is documented; these results evaluate the
          committed weights rather than claiming an optimal training run.
        </p>
        <p>
          Dataset:{" "}
          <a
            href="https://huggingface.co/datasets/tarekmasryo/digital-lifestyle-benchmark-dataset"
            target="_blank"
            rel="noopener noreferrer"
          >
            Digital Lifestyle Benchmark (2025), Tarek Masryo
          </a>
          ,{" "}
          <a
            href="https://creativecommons.org/licenses/by/4.0/"
            target="_blank"
            rel="noopener noreferrer"
          >
            CC BY 4.0
          </a>
          . This demo uses derived evaluation results and fictional example
          profiles.{" "}
          <a
            href={`${source}/blob/showcase/model-explorer/docs/MODEL_CARD.md`}
            target="_blank"
            rel="noopener noreferrer"
          >
            Read the model card ↗
          </a>
        </p>
      </section>
    </>
  );
}

function FeatureControl({
  feature,
  value,
  onChange,
  onValidity,
}: {
  feature: Feature;
  value: number;
  onChange: (value: number) => void;
  onValidity: (valid: boolean) => void;
}) {
  const [draft, setDraft] = useState(String(value));
  const valid =
    draft.trim() !== "" &&
    Number.isFinite(Number(draft)) &&
    Number(draft) >= feature.min &&
    Number(draft) <= feature.max;
  const id = `feature-${feature.key}`;
  return (
    <div className="feature">
      <div className="feature-heading">
        <label htmlFor={id}>{feature.label}</label>
        <input
          className="number-input"
          id={id}
          type="number"
          min={feature.min}
          max={feature.max}
          step="any"
          value={draft}
          aria-invalid={!valid}
          aria-describedby={`${id}-unit${valid ? "" : ` ${id}-error`}`}
          onChange={(event) => {
            const next = event.target.value;
            setDraft(next);
            const ok =
              next.trim() !== "" &&
              Number.isFinite(Number(next)) &&
              Number(next) >= feature.min &&
              Number(next) <= feature.max;
            onValidity(ok);
            if (ok) onChange(Number(next));
          }}
        />
      </div>
      <input
        type="range"
        aria-label={`${feature.label} slider`}
        min={feature.min}
        max={feature.max}
        step="any"
        value={value}
        onChange={(event) => {
          const next = Math.max(
            feature.min,
            Math.min(
              feature.max,
              Number(Number(event.target.value).toFixed(feature.decimals)),
            ),
          );
          setDraft(String(next));
          onValidity(true);
          onChange(next);
        }}
      />
      <div className="feature-meta" id={`${id}-unit`}>
        <span>{feature.unit}</span>
        <span>
          {format(feature.min, feature.decimals)}–
          {format(feature.max, feature.decimals)}
        </span>
      </div>
      {!valid && (
        <small id={`${id}-error`} className="field-error">
          Enter a value within the displayed range.
        </small>
      )}
    </div>
  );
}

function App() {
  const [release, setRelease] = useState<Release | null>(null);
  const [failed, setFailed] = useState(false);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setFailed(false);
    fetch("/release.json", { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error("Metadata unavailable");
        return response.json();
      })
      .then(setRelease)
      .catch((error) => {
        if (error.name !== "AbortError") setFailed(true);
      });
    return () => controller.abort();
  }, [attempt]);
  return (
    <>
      <a className="skip-link" href="#explore">
        Skip to explorer
      </a>
      <header className="site-header">
        <a href="#" className="brand" aria-label="MindMetrics home">
          <Mark />
          <span>
            MindMetrics<span className="brand-suffix"> / ML</span>
          </span>
        </a>
        <nav aria-label="Main navigation">
          <a href="#explore">Explorer</a>
          <a href="#evaluation">Evaluation</a>
          <a
            href={`${source}/tree/showcase/model-explorer`}
            target="_blank"
            rel="noopener noreferrer"
          >
            GitHub ↗
          </a>
        </nav>
      </header>
      <main>
        <section className="hero">
          <div className="hero-copy">
            <p className="eyebrow">
              <span className="status-dot" />
              AN INTERACTIVE MACHINE LEARNING STUDY
            </p>
            <h1>
              The model,
              <br />
              made <em>tangible.</em>
            </h1>
            <p className="hero-description">
              Everyday habits. Learned patterns. Explore how a neural network
              turns fifteen inputs into one prediction.
            </p>
            <div className="hero-actions">
              <a className="primary-button" href="#explore">
                Explore the model <span aria-hidden="true">↘</span>
              </a>
              <span>
                Fictional profiles.
                <br />
                Real inference.
              </span>
            </div>
          </div>
          <div className="hero-art" aria-hidden="true">
            <div className="art-label">INPUTS → PATTERNS → PREDICTION</div>
            <svg viewBox="0 0 480 350">
              <defs>
                <radialGradient id="glow">
                  <stop stopColor="#dce8ba" />
                  <stop offset="1" stopColor="#e9e9dd" stopOpacity="0" />
                </radialGradient>
              </defs>
              <circle cx="250" cy="175" r="170" fill="url(#glow)" />
              {[65, 125, 185, 245].flatMap((y, i) =>
                [90, 170, 250].flatMap((z, j) =>
                  [0, 1, 2].map((k) => (
                    <path
                      key={`${i}-${j}-${k}`}
                      d={`M${80 + j * 115} ${y + j * 10}L${195 + j * 115} ${65 + k * 80}`}
                      stroke="#5c7d64"
                      strokeOpacity=".22"
                    />
                  )),
                ),
              )}
              {[80, 195, 310].flatMap((x, j) =>
                [0, 1, 2, 3].map((k) => (
                  <g key={`${j}-${k}`}>
                    <circle
                      cx={x}
                      cy={65 + k * 60 + j * 10}
                      r={j === 1 ? 12 : 8}
                      fill={j === 1 ? "#254e43" : "#a4b594"}
                    />
                    <circle
                      cx={x}
                      cy={65 + k * 60 + j * 10}
                      r="3"
                      fill="#ecf1d4"
                    />
                  </g>
                )),
              )}
              <path
                d="M320 105L420 175L320 245"
                fill="none"
                stroke="#5c7d64"
                strokeOpacity=".4"
              />
              <circle cx="420" cy="175" r="24" fill="#254e43" />
              <path
                d="M408 175h24m-12-12v24"
                stroke="#e6f3bc"
                strokeWidth="2"
              />
            </svg>
            <div className="art-foot">
              <span>15 FEATURES</span>
              <span>RESIDUAL NETWORK</span>
              <span>1 SCORE</span>
            </div>
          </div>
        </section>
        <div className="trust-strip">
          <span>
            <i /> Runs entirely in your browser
          </span>
          <span>Built on 3,500 synthetic records</span>
          <span>No account. No input collection.</span>
        </div>
        {release ? (
          <Explorer release={release} />
        ) : (
          <section className="load-panel" role="status">
            {failed ? (
              <>
                <p>We couldn’t load the model’s release data.</p>
                <button onClick={() => setAttempt((a) => a + 1)}>
                  Try again
                </button>
              </>
            ) : (
              "Loading the explorer…"
            )}
          </section>
        )}
      </main>
      <footer>
        <a href="https://sanhithamarathunge.net/projects#project-mindmetrics">
          ← Back to Sanhith’s portfolio
        </a>
        <span>MindMetrics-ML · Academic project, thoughtfully revisited.</span>
        <a href={`${source}/blob/showcase/model-explorer/docs/REVIEW.md`}>
          Release notes ↗
        </a>
      </footer>
    </>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
