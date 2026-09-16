// cetk_fixgw - FIX 4.4 RFQ gateway in front of the engine, and the identical-fills self-test.
//   cetk_fixgw --serve   [--root CL] [--port 9879] [--days 3] [--quoter BEGV_adv_cal] [--hedger ImpactAware]
//   cetk_fixgw --selftest [--root CL] [--port 9879] [--days 3]
//        replays the same RFQ script in-process and through a FIX session (gateway thread + client thread on
//        localhost) and asserts that the fill lists (time, tenor, side, size, price) and the final P&L are identical.
#include "cetk/fix.hpp"
#include <thread>
#include <iostream>
#include <map>
#include <memory>

using namespace cetk;

static std::map<std::string, std::string> parse(int argc, char** argv) { std::map<std::string, std::string> kv; for (int i = 1; i < argc; ++i) { std::string s = argv[i]; if (s.rfind("--", 0) == 0) { std::string k = s.substr(2); if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) kv[k] = argv[++i]; else kv[k] = "1"; } } return kv; }
static double num(const std::map<std::string, std::string>& kv, const std::string& k, double d) { auto it = kv.find(k); return it == kv.end() ? d : std::atof(it->second.c_str()); }
static std::string str(const std::map<std::string, std::string>& kv, const std::string& k, const std::string& d) { auto it = kv.find(k); return it == kv.end() ? d : it->second; }
static std::string lower(std::string s) { for (auto& c : s) c = (char)std::tolower(c); return s; }

static FactorModel factors_from_json(const Json& j) { FactorModel f; for (auto& v : j["months"].arr()) f.months.push_back((int)v.num()); for (auto& v : j["evals_usd2_per_day"].arr()) f.evals.push_back(v.num()); for (auto& r : j["loadings"].arr()) { Vec row; for (auto& v : r.arr()) row.push_back(v.num()); f.loadings.push_back(row); } f.K = (int)f.loadings.size(); return f; }

struct Setup { EngineConfig cfg; std::vector<DaySpec> days; std::unique_ptr<Quoter> quoter; };
static Setup make(const std::map<std::string, std::string>& kv) {
    Setup S; std::string root = str(kv, "root", "CL"); LadderSpec ls; ls.root = root; ls.contract_size = root == "NG" ? 10000 : 1000; ls.price_level = root == "NG" ? 3.0 : 70.0;
    FactorModel fm = factors_from_json(Json::parse(read_file(str(kv, "factors", "results/factors_" + lower(root) + ".json")))["recent"]);
    ImpactModel im = ImpactModel::from_json(Json::parse(read_file(str(kv, "impact", "configs/impact_" + lower(root) + ".json"))));
    LadderRisk risk; risk.build(fm, ls); auto tiers = default_tiers(root); if (root == "NG") for (auto& t : tiers) { t.beta *= 20; t.mu /= 20; }
    QuoteParams qp; qp.gamma = num(kv, "gamma", 5e-7); Hedger H; H.impact = im; H.Sigma = risk.Sigma; H.contract_size = ls.contract_size; H.gamma = qp.gamma; H.tau_days = 0.25; H.policy = parse_hedge(str(kv, "hedger", "ImpactAware"));
    S.cfg.ls = ls; S.cfg.qp = qp; S.cfg.tiers = tiers; S.cfg.risk = risk; S.cfg.hedger = H; S.cfg.market_seed = (uint64_t)num(kv, "seed", 1); S.cfg.F0.assign(ls.months.size(), ls.price_level); S.cfg.info_drift_per_day = root == "NG" ? 0.03 : 0.6;
    Rng drng(77); S.days = synthetic_days((int)num(kv, "days", 3), risk, drng, 0.5, 2.0);
    std::string qn = str(kv, "quoter", "BEGV_adv_cal"); if (qn == "Symmetric") S.quoter = std::make_unique<SymmetricQuoter>(tiers, qp); else if (qn == "BEGV") S.quoter = std::make_unique<BegvQuoter>(tiers, ls, risk, qp, false, false); else if (qn == "Oracle") S.quoter = std::make_unique<BegvQuoter>(tiers, ls, risk, qp, true, true); else S.quoter = std::make_unique<BegvQuoter>(tiers, ls, risk, qp, true, false);
    return S;
}

static std::string fills_key(const std::vector<Fill>& fs) { std::string k; char b[64]; for (auto& f : fs) { std::snprintf(b, sizeof b, "%.6f:%d:%d:%g:%.8f;", f.t, f.tenor, f.side, f.z, f.mid + f.side * f.delta); k += b; } return k; }
static std::string fills_key(const std::vector<fix::ClientFill>& fs, double day_s) { std::string k; char b[64]; size_t day = 0; double last_t = -1; for (auto& f : fs) { if (f.t < last_t) ++day; last_t = f.t; std::snprintf(b, sizeof b, "%.6f:%d:%d:%g:%.8f;", f.t + day * day_s, f.tenor, f.side, f.z, f.price); k += b; } return k; }

int main(int argc, char** argv) {
    auto kv = parse(argc, argv); int port = (int)num(kv, "port", 9879);
    try {
        if (kv.count("serve")) {
            Setup S = make(kv); RfqEngine eng(S.cfg); eng.set_quoter(S.quoter.get());
            sock_t ls = fix::listen_on(port); std::cerr << "[cetk_fixgw] listening on 127.0.0.1:" << port << std::endl; sock_t cs = ::accept(ls, nullptr, nullptr); fix::Conn conn(cs); fix::RfqGateway gw(eng, S.days); gw.serve(conn);
            std::cerr << "[cetk_fixgw] session ended: " << eng.stats().n_fills << " fills, P&L " << eng.pnl_now() << std::endl; return 0;
        }
        // ---- self-test ------------------------------------------------------------------------------------------
        int failures = 0;
        for (std::string qn : {"Symmetric", "BEGV_adv_cal"}) {
            auto kv2 = kv; kv2["quoter"] = qn;
            Setup in = make(kv2); RfqEngine eng_in(in.cfg); eng_in.set_quoter(in.quoter.get()); Rng crng(123);
            for (size_t d = 0; d < in.days.size(); ++d) { eng_in.start_day(in.days[d]); auto script = generate_flow(in.cfg, in.days[d], eng_in.flow_dir(), crng); for (auto& r : script) { eng_in.advance(r.t); auto q = eng_in.quote(r.tier, r.tenor, r.side, r.z); if (!q.valid) continue; if (r.u < in.cfg.tiers[r.tier].f(q.delta)) eng_in.execute(q); } eng_in.end_day(); }
            Setup gw = make(kv2); RfqEngine eng_gw(gw.cfg); eng_gw.set_quoter(gw.quoter.get());
            sock_t ls = fix::listen_on(port); std::thread server([&]() { sock_t cs = ::accept(ls, nullptr, nullptr); fix::Conn conn(cs); fix::RfqGateway g(eng_gw, gw.days); g.serve(conn); });
            std::vector<fix::ClientFill> fills; size_t n_quotes = 0;
            { fix::Conn conn(fix::connect_to(port)); fix::RfqClient client(conn); Rng crng2(123); Setup cl = make(kv2);
              for (size_t d = 0; d < cl.days.size(); ++d) { double dir = client.start_day(d); auto script = generate_flow(cl.cfg, cl.days[d], dir, crng2); for (auto& r : script) client.rfq(r, cl.cfg.tiers[r.tier], cl.cfg.ls.root); client.end_day(); }
              fills = client.fills; n_quotes = client.n_quotes_; client.logout(); }
            server.join(); CLOSESOCK(ls);
            std::string k_in = fills_key(eng_in.stats().fills), k_fix = fills_key(fills, in.cfg.day_s()); bool same = k_in == k_fix && std::fabs(eng_in.pnl_now() - eng_gw.pnl_now()) < 1e-6;
            std::printf("%-14s in-process: %4zu fills, P&L %12.2f | FIX: %4zu fills over %zu quotes, P&L %12.2f | %s\n", qn.c_str(), eng_in.stats().fills.size(), eng_in.pnl_now(), fills.size(), n_quotes, eng_gw.pnl_now(), same ? "IDENTICAL" : "MISMATCH");
            if (!same) ++failures;
        }
        std::printf("%s\n", failures ? "FIX SELF-TEST FAILED" : "FIX SELF-TEST PASSED: identical fills and P&L in-process and through FIX 4.4"); return failures ? 1 : 0;
    } catch (const std::exception& e) { std::cerr << "error: " << e.what() << std::endl; return 1; }
}
