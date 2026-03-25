import "katex/dist/katex.min.css";
import { InlineMath, BlockMath } from "react-katex";

export function InlineEquation({ math }) {
  return <InlineMath math={math} />;
}

export function DisplayEquation({ math, label }) {
  return (
    <div className="my-4 overflow-x-auto">
      <BlockMath math={math} />
      {label && <span className="text-xs text-light/40 mt-1 block text-right">{label}</span>}
    </div>
  );
}

export default DisplayEquation;
