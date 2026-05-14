import Link from 'next/link';
import { Scale, ShieldCheck, Clock, BrainCircuit, ArrowRight, Gavel, Scale as ScaleIcon } from 'lucide-react';

export default function Home() {
  return (
    <div className="min-h-screen bg-[#ede8de] text-[#2d1f0e] font-sans overflow-x-hidden selection:bg-[#c8b89a] selection:text-[#2d1f0e]">
      
      {/* Decorative Background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
        <div className="absolute -top-[20%] -right-[10%] w-[60%] h-[70%] rounded-full bg-[#74603e] opacity-5 blur-[120px]" />
        <div className="absolute -bottom-[20%] left-[10%] w-[50%] h-[60%] rounded-full bg-[#9e8453] opacity-10 blur-[100px]" />
        <div className="absolute inset-0 opacity-[0.03]" style={{ backgroundImage: 'radial-gradient(#2d1f0e 1px, transparent 1px)', backgroundSize: '24px 24px' }} />
      </div>

      {/* Navigation */}
      <nav className="relative z-10 w-full px-6 py-6 sm:px-12 lg:px-20 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-[#74603e]/10 rounded-xl border border-[#c8b89a] shadow-sm">
            <ScaleIcon className="w-6 h-6 text-[#74603e]" />
          </div>
          <span className="text-xl font-bold text-[#2d1f0e] tracking-tight">Family Law AI</span>
        </div>
        <div className="flex items-center gap-4">
          <Link href="/auth" className="text-sm font-medium text-[#4a3728] hover:text-[#74603e] transition-colors">
            Sign In
          </Link>
          <Link href="/auth" className="text-sm font-medium bg-[#74603e] text-white px-5 py-2.5 rounded-xl hover:bg-[#5c4b2f] transition-all shadow-md hover:shadow-lg flex items-center gap-2">
            Get Started <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </nav>

      {/* Hero Section */}
      <main className="relative z-10 flex flex-col items-center justify-center pt-24 pb-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto text-center">
        
       

        <h1 className="text-5xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight mb-8 text-[#2d1f0e] leading-[1.1]">
          Navigate Family Law <br className="hidden sm:block" />
          <span className="text-[#74603e] inline-block mt-2">With Confidence</span>
        </h1>

        <p className="max-w-2xl text-lg sm:text-xl text-[#4a3728] mb-12 leading-relaxed mx-auto">
          Clear, explainable, and empathetic legal guidance for your most sensitive matters. Instant AI insights backed by actual legal precedents, tailored specifically to your situation.
        </p>

        <div className="flex flex-col sm:flex-row gap-4 items-center justify-center">
          <Link 
            href="/auth" 
            className="w-full sm:w-auto bg-[#74603e] text-white px-8 py-4 rounded-2xl text-lg font-semibold hover:bg-[#5c4b2f] hover:scale-105 transition-all duration-300 shadow-xl shadow-[#74603e]/20 flex items-center justify-center gap-3"
          >
            Start Your Consultation <ArrowRight className="w-5 h-5" />
          </Link>
        </div>

        {/* Features Grid */}
        <div className="mt-32 grid grid-cols-1 md:grid-cols-3 gap-8 w-full max-w-5xl">
          <FeatureCard 
            icon={<ShieldCheck className="w-6 h-6 text-[#74603e]" />}
            title="Private & Secure"
            description="Your sensitive family matters are discussed in a completely confidential and secure environment."
          />
          <FeatureCard 
            icon={<BrainCircuit className="w-6 h-6 text-[#74603e]" />}
            title="Explainable Reasoning"
            description="Not just answers. We provide step-by-step reasoning and cite exact legal precedents for transparency."
          />
          <FeatureCard 
            icon={<Clock className="w-6 h-6 text-[#74603e]" />}
            title="Available 24/7"
            description="Get immediate legal insights and prepare for your case anytime, anywhere, without the wait."
          />
        </div>

      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-[#c8b89a] mt-20 bg-[#e2dbd0]/50 py-8 text-center text-[#8a7462] text-sm">
        <p>© {new Date().getFullYear()} Family Law Assistant AI. All rights reserved.</p>
        <p className="mt-2 text-xs">This tool provides legal information and analysis, not official legal advice.</p>
      </footer>

    </div>
  );
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode, title: string, description: string }) {
  return (
    <div className="bg-white/80 backdrop-blur-md rounded-3xl p-8 text-left border border-[#c8b89a] shadow-[0_8px_30px_rgba(0,0,0,0.04)] hover:shadow-[0_8px_30px_rgba(116,96,62,0.12)] transition-all duration-300 hover:-translate-y-1">
      <div className="w-12 h-12 rounded-2xl bg-[#f7f3ec] border border-[#c8b89a] flex items-center justify-center mb-6">
        {icon}
      </div>
      <h3 className="text-xl font-bold text-[#2d1f0e] mb-3">{title}</h3>
      <p className="text-[#4a3728] leading-relaxed">{description}</p>
    </div>
  );
}