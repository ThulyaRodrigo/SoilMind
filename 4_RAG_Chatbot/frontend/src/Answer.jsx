import React from "react";

// Renders the model's plain text: paragraphs, "- " lists, **bold**, and [n] citations.
function inline(text, keyBase) {
  return text.split(/(\*\*[^*]+\*\*|\[\d+\])/g).map((part, i) => {
    if (/^\*\*[^*]+\*\*$/.test(part)) return <strong key={keyBase + i}>{part.slice(2, -2)}</strong>;
    if (/^\[\d+\]$/.test(part))
      return (
        <sup key={keyBase + i} className="cite" title={`Source ${part.slice(1, -1)}`}>
          {part.slice(1, -1)}
        </sup>
      );
    return part;
  });
}

function blocks(text) {
  const out = [];
  let list = null;
  text.split("\n").forEach((raw) => {
    const line = raw.trim();
    const item = line.match(/^([-*•]|\d+[.)])\s+(.*)$/);
    if (item) {
      if (!list) {
        list = { type: "ul", items: [] };
        out.push(list);
      }
      list.items.push(item[2]);
    } else if (line) {
      list = null;
      out.push({ type: "p", text: line });
    } else {
      list = null;
    }
  });
  return out;
}

export default function Answer({ text, sources = [] }) {
  return (
    <div className="answer">
      {blocks(text).map((b, i) =>
        b.type === "ul" ? (
          <ul key={i}>
            {b.items.map((it, j) => (
              <li key={j}>{inline(it, `${i}-${j}-`)}</li>
            ))}
          </ul>
        ) : (
          <p key={i}>{inline(b.text, `${i}-`)}</p>
        )
      )}
      {sources.length > 0 && (
        <details className="sources">
          <summary>Sources ({sources.length})</summary>
          <ol>
            {sources.map((s) => (
              <li key={s.id} value={s.id}>
                <span className="src-title">{s.title}</span>
                <span className="src-file">{s.file}</span>
              </li>
            ))}
          </ol>
        </details>
      )}
    </div>
  );
}
