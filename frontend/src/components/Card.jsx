export default function Card({ variant = "default", children, className = "", ...props }) {
  const base = "rounded-card p-6 border transition-shadow duration-200 hover:shadow-md";

  const variants = {
    default: "bg-white border-divider",
    pcos: "bg-blush border-soft-rose",
    endo: "bg-endo-light border-endo/40",
    ai: "bg-ai-light border-ai/40",
  };

  return (
    <div className={`${base} ${variants[variant]} ${className}`} {...props}>
      {children}
    </div>
  );
}