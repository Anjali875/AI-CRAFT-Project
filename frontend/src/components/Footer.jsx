import { Link } from "react-router-dom";
import { Heart } from "lucide-react";

export default function Footer() {
  return (
    <footer className="bg-blush border-t border-divider mt-16">
      <div className="max-w-6xl mx-auto px-6 py-12 grid grid-cols-1 md:grid-cols-5 gap-8">
        <div className="md:col-span-2">
          <div className="flex items-center gap-2 mb-3">
            <span className="w-8 h-8 rounded-full bg-pcos flex items-center justify-center">
              <Heart size={16} className="text-white" fill="white" />
            </span>
            <span className="font-heading font-semibold text-charcoal">Women's Health AI</span>
          </div>
          <p className="text-sm text-body max-w-xs">
            Empowering women with AI-driven insights for a healthier tomorrow.
          </p>
        </div>

        <div>
          <h4 className="font-heading font-semibold text-charcoal text-sm mb-3">Quick Links</h4>
          <ul className="space-y-2 text-sm text-body">
            <li><Link to="/" className="hover:text-pcos">Home</Link></li>
            <li><Link to="/screening" className="hover:text-pcos">Screening</Link></li>
            <li><Link to="/health-library" className="hover:text-pcos">Health Library</Link></li>
            <li><Link to="/about" className="hover:text-pcos">About Us</Link></li>
          </ul>
        </div>

        <div>
          <h4 className="font-heading font-semibold text-charcoal text-sm mb-3">Resources</h4>
          <ul className="space-y-2 text-sm text-body">
            <li><span className="hover:text-pcos cursor-pointer">PCOS Guide</span></li>
            <li><span className="hover:text-pcos cursor-pointer">Endometriosis Guide</span></li>
            <li><span className="hover:text-pcos cursor-pointer">Symptom Checker</span></li>
            <li><span className="hover:text-pcos cursor-pointer">FAQ</span></li>
          </ul>
        </div>

        <div>
          <h4 className="font-heading font-semibold text-charcoal text-sm mb-3">Legal</h4>
          <ul className="space-y-2 text-sm text-body">
            <li><span className="hover:text-pcos cursor-pointer">Privacy Policy</span></li>
            <li><span className="hover:text-pcos cursor-pointer">Terms of Use</span></li>
            <li><span className="hover:text-pcos cursor-pointer">Disclaimer</span></li>
          </ul>
        </div>
      </div>
      <div className="border-t border-divider py-4 text-center text-xs text-body">
        © 2026 Women's Health AI. All rights reserved.
      </div>
    </footer>
  );
}