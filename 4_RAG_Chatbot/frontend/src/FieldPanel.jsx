import React from "react";

const CROPS = ["Rice", "Vegetables", "Tea", "Coconut", "Other"];

const FIELDS = [
  { key: "ph", label: "Soil pH", unit: "", step: "0.1", min: 0, max: 14 },
  { key: "moisture", label: "Moisture", unit: "%", step: "1", min: 0, max: 100 },
  { key: "n", label: "Nitrogen (N)", unit: "mg/kg", step: "1", min: 0 },
  { key: "p", label: "Phosphorus (P)", unit: "mg/kg", step: "1", min: 0 },
  { key: "k", label: "Potassium (K)", unit: "mg/kg", step: "1", min: 0 },
];

export default function FieldPanel({ soil, onChange, onClear }) {
  const filled = Object.values(soil).some((v) => v !== "");
  return (
    <div className="panel-body">
      <p className="panel-help">
        Add your sensor or test readings. They are sent with every question so answers fit your field.
      </p>

      <label className="field">
        <span>Crop</span>
        <select value={soil.crop} onChange={(e) => onChange("crop", e.target.value)}>
          <option value="">Not set</option>
          {CROPS.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </label>

      {FIELDS.map((f) => (
        <label className="field" key={f.key}>
          <span>{f.label}</span>
          <div className="num">
            <input
              type="number"
              inputMode="decimal"
              step={f.step}
              min={f.min}
              max={f.max}
              value={soil[f.key]}
              onChange={(e) => onChange(f.key, e.target.value)}
            />
            {f.unit && <em>{f.unit}</em>}
          </div>
        </label>
      ))}

      <button type="button" className="link-btn" onClick={onClear} disabled={!filled}>
        Clear readings
      </button>
    </div>
  );
}
