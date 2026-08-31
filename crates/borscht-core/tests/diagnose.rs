use borscht_core::fastmath::gaussian;
use borscht_core::genome::{self, pg};
use borscht_core::{Config, World};

#[test]
fn breakdown() {
    let mut c = Config::for_population(60_000);
    c.sanitize();
    let mut w = World::new(c, 1);
    let t = genome::tables();
    for step in [0usize, 400] {
        if step > 0 { w.tick_many(step as u32); }
        let (mut light, mut fit, mut shade, mut uptake, mut head, mut rate, mut growth, mut bm) =
            (0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
        let n = w.plants.len();
        for i in 0..n {
            let cell = w.grid.cell_of(w.plants.x[i], w.plants.y[i]) as usize;
            let row = w.grid.geom.row_of(cell as u32) as usize;
            let b_tol = w.plants.gene(i, pg::TEMP_TOLERANCE) as usize;
            let f = t.plant_temp_peak[b_tol]
                * gaussian(w.env.row_temp[row] - t.plant[pg::TEMP_OPT][w.plants.gene(i, pg::TEMP_OPT) as usize],
                           t.plant[pg::TEMP_TOLERANCE][b_tol]);
            let s = w.grid.soil[cell];
            let sh = w.cfg.shade_half / (w.cfg.shade_half + w.grid.plant_mass[cell]);
            let up = s / (s + w.cfg.soil_half);
            let max = t.plant[pg::MAX_SIZE][w.plants.gene(i, pg::MAX_SIZE) as usize];
            let hd = (1.0 - w.plants.biomass[i] / max).clamp(0.0, 1.0);
            let r = t.plant[pg::GROWTH_RATE][w.plants.gene(i, pg::GROWTH_RATE) as usize];
            let l = w.env.row_light[row];
            light += l as f64; fit += f as f64; shade += sh as f64; uptake += up as f64;
            head += hd as f64; rate += r as f64; bm += (w.plants.biomass[i] / max) as f64;
            growth += (r * w.plants.biomass[i] * l * f * sh * up * hd) as f64;
        }
        let d = n as f64;
        println!("tick {:>4} n={} light={:.3} fit={:.3} shade={:.3} uptake={:.3} headroom={:.3} rate={:.4} | growth/tick={:.4} b/max={:.3}",
            w.tick, n, light/d, fit/d, shade/d, uptake/d, head/d, rate/d, growth/d, bm/d);
    }
}
