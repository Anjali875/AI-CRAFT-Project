function renderInline(text, kp) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((p, i) =>
    p.startsWith("**") && p.endsWith("**") ? (
      <strong key={`${kp}-b${i}`}>{p.slice(2, -2)}</strong>
    ) : (
      <span key={`${kp}-t${i}`}>{p}</span>
    )
  );
}

export default function FormattedMessage({ text }) {
  const lines = text.split("\n");
  const blocks = [];
  let listItems = [];
  const flush = (key) => {
    if (listItems.length) {
      blocks.push(
        <ul key={`ul-${key}`} className="list-disc list-inside space-y-1 my-1">
          {listItems}
        </ul>
      );
      listItems = [];
    }
  };
  lines.forEach((line, i) => {
    const t = line.trim();
    if (!t) return flush(i);
    if (/^#{1,6}\s/.test(t)) {
      flush(i);
      blocks.push(
        <p key={`h-${i}`} className="font-semibold text-charcoal mt-2">
          {renderInline(t.replace(/^#{1,6}\s/, ""), `h${i}`)}
        </p>
      );
    } else if (/^[-*]\s/.test(t)) {
      listItems.push(<li key={`li-${i}`}>{renderInline(t.replace(/^[-*]\s/, ""), `li${i}`)}</li>);
    } else {
      flush(i);
      blocks.push(<p key={`p-${i}`} className="my-1">{renderInline(t, `p${i}`)}</p>);
    }
  });
  flush("end");
  return <div className="text-sm leading-relaxed">{blocks}</div>;
}