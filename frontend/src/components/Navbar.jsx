import { useState } from "react";
import { NavLink, Link, useNavigate } from "react-router-dom";
import { Heart, MessageCircle, Menu, X } from "lucide-react";

export default function Navbar() {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);

  const links = [
    { to: "/", label: "Home" },
    { to: "/screening", label: "Screening" },
    { to: "/health-library", label: "Health Library" },
    { to: "/about", label: "About" },
  ];

  return (
    <nav className="sticky top-0 z-50 bg-white/90 backdrop-blur border-b border-divider shadow-sm">
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link to="/" onClick={() => setOpen(false)} className="flex items-center gap-2">
          <span className="w-9 h-9 rounded-full bg-pcos flex items-center justify-center">
            <Heart size={18} className="text-white" fill="white" />
          </span>
          <span className="font-heading font-semibold text-lg text-charcoal">
            Women's Health AI
          </span>
        </Link>

        {/* Desktop links */}
        <div className="hidden md:flex items-center gap-8">
          {links.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              end={l.to === "/"}
              className={({ isActive }) =>
                `text-sm font-medium transition-colors ${
                  isActive ? "text-pcos" : "text-charcoal hover:text-pcos"
                }`
              }
            >
              {l.label}
            </NavLink>
          ))}
        </div>

        {/* Desktop AI Assistant button */}
        <button
          onClick={() => navigate("/assistant")}
          className="hidden md:inline-flex items-center gap-2 rounded-pill border border-ai text-ai text-sm font-medium px-4 py-2 hover:bg-ai-light transition-colors cursor-pointer"
        >
          <MessageCircle size={16} /> AI Assistant
        </button>

        {/* Mobile hamburger */}
        <button
          onClick={() => setOpen((v) => !v)}
          aria-label={open ? "Close menu" : "Open menu"}
          aria-expanded={open}
          className="md:hidden inline-flex items-center justify-center w-10 h-10 rounded-xl text-charcoal hover:text-pcos transition-colors cursor-pointer"
        >
          {open ? <X size={22} /> : <Menu size={22} />}
        </button>
      </div>

      {/* Mobile panel */}
      {open && (
        <div className="md:hidden border-t border-divider bg-white px-6 py-4 flex flex-col gap-1">
          {links.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              end={l.to === "/"}
              onClick={() => setOpen(false)}
              className={({ isActive }) =>
                `text-base font-medium py-3 transition-colors ${
                  isActive ? "text-pcos" : "text-charcoal hover:text-pcos"
                }`
              }
            >
              {l.label}
            </NavLink>
          ))}
          <button
            onClick={() => {
              setOpen(false);
              navigate("/assistant");
            }}
            className="mt-2 inline-flex items-center justify-center gap-2 rounded-pill border border-ai text-ai text-sm font-medium px-4 py-3 hover:bg-ai-light transition-colors cursor-pointer"
          >
            <MessageCircle size={16} /> AI Assistant
          </button>
        </div>
      )}
    </nav>
  );
}