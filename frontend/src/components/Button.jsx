export default function Button({ variant = "primary", children, className = "", ...props }) {
  const base =
    "inline-flex items-center justify-center gap-2 rounded-pill font-medium px-6 py-3 transition-colors duration-200 cursor-pointer";

  const variants = {
    primary: "bg-primary text-white hover:bg-deep-rose",
    secondary: "bg-transparent text-primary border border-primary hover:bg-blush",
  };

  return (
    <button className={`${base} ${variants[variant]} ${className}`} {...props}>
      {children}
    </button>
  );
}