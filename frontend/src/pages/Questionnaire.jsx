import { useParams } from "react-router-dom";
import PCOSQuestionnaire from "./PCOSQuestionnaire";
import EndoQuestionnaire from "./EndoQuestionnaire";

export default function Questionnaire() {
  const { condition } = useParams();
  if (condition === "pcos") return <PCOSQuestionnaire />;
  if (condition === "endo") return <EndoQuestionnaire />;
  return (
    <div className="max-w-2xl mx-auto px-6 py-16 text-center">
      <h1 className="text-2xl text-charcoal">Unknown screening type.</h1>
    </div>
  );
}