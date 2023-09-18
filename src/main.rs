use ggez::filesystem::print_all;
use rand::Rng;
use std::fmt;

// grid info
const GRID_SIZE: usize = 80;
const CELL_SIZE: f32 = 50.0;

// gene encoding
const GENE_SIZE: usize = 32; // bits per gene
const CONNECTION_BITS: usize = 1; // First bit for positive/negative, change as needed
const SOURCE_BITS: usize = 8; // Number of bits for source neuron
const DESTINATION_BITS: usize = 8; // Number of bits for destination neuron
const WEIGHT_BITS: usize = GENE_SIZE - CONNECTION_BITS - SOURCE_BITS - DESTINATION_BITS; // remaining bits for weight

const NUM_GENES: usize = 1;

enum inputNeurons {
    BorderDistanceTop,
    BorderDistanceBot,
    BorderDistanceRight,
    BorderDistanceLeft,
    countNeighbors,
    Age,
}
enum outputNeurons {
    MvUp,
    MvD,
    MvR,
    MvL,
    MvRandom,
    Halt,
}

impl fmt::Display for inputNeurons {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            inputNeurons::BorderDistanceTop => write!(f, "BorderDistanceTop"),
            inputNeurons::BorderDistanceBot => write!(f, "BorderDistanceBot"),
            inputNeurons::BorderDistanceRight => write!(f, "BorderDistanceRight"),
            inputNeurons::BorderDistanceLeft => write!(f, "BorderDistanceLeft"),
            inputNeurons::countNeighbors => write!(f, "countNeighbors"),
            inputNeurons::Age => write!(f, "Age"),
        }
    }
}

impl fmt::Display for outputNeurons {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            outputNeurons::MvUp => write!(f, "MvUp"),
            outputNeurons::MvD => write!(f, "MvD"),
            outputNeurons::MvR => write!(f, "MvR"),
            outputNeurons::MvL => write!(f, "MvL"),
            outputNeurons::MvRandom => write!(f, "MvRandom"),
            outputNeurons::Halt => write!(f, "Halt"),
        }
    }
}

const NUM_INPUT_NEURONS: usize = 6;
const NUM_OUTPUT_NEURONS: usize = 6;

impl inputNeurons {
    fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(inputNeurons::BorderDistanceTop),
            1 => Some(inputNeurons::BorderDistanceBot),
            2 => Some(inputNeurons::BorderDistanceRight),
            3 => Some(inputNeurons::BorderDistanceLeft),
            4 => Some(inputNeurons::countNeighbors),
            5 => Some(inputNeurons::Age),
            _ => None,
        }
    }
}

impl outputNeurons {
    fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(outputNeurons::MvUp),
            1 => Some(outputNeurons::MvD),
            2 => Some(outputNeurons::MvR),
            3 => Some(outputNeurons::MvL),
            4 => Some(outputNeurons::MvRandom),
            5 => Some(outputNeurons::Halt),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
struct Gene {
    value: u32,
}

impl Gene {
    fn newRandom() -> Self {
        let mut rng = rand::thread_rng();
        let value = rng.gen_range(0..=u32::MAX);
        Gene { value }
    }

    fn is_positive(&self) -> bool {
        (self.value >> (GENE_SIZE - CONNECTION_BITS - 1)) & 1 == 0
    }

    fn source_neuron(&self) -> Option<inputNeurons> {
        let source_index = (self.value >> (GENE_SIZE - CONNECTION_BITS - SOURCE_BITS - 1))
            & ((1 << SOURCE_BITS) - 1);
        let mapped_source_index = ((source_index as f32 / 255.0) * 6.0) as usize;

        inputNeurons::from_index(mapped_source_index)
    }

    fn destination_neuron(&self) -> Option<outputNeurons> {
        let dest_index = (self.value
            >> (GENE_SIZE - CONNECTION_BITS - SOURCE_BITS - DESTINATION_BITS - 1))
            & ((1 << DESTINATION_BITS) - 1);

        let mapped_dest_index = ((dest_index as f32 / 255.0) * 6.0) as usize;

        outputNeurons::from_index(mapped_dest_index)
    }

    fn weight(&self) -> f32 {
        let weight_bits = (self.value & ((1 << WEIGHT_BITS) - 1)) as f32;
        if self.is_positive() {
            weight_bits / (1 << WEIGHT_BITS) as f32 * 4.0
        } else {
            -(weight_bits / (1 << WEIGHT_BITS) as f32 * 4.0)
        }
    }
}

#[derive(Clone)]
struct Bob {
    position_x: f32,
    position_y: f32,
    genes: Vec<Gene>,
    Age: f32,
}

impl Bob {
    fn new(x: usize, y: usize) -> Self {
        let genes = vec![
            Gene::newRandom(),
            Gene::newRandom(),
            Gene::newRandom(),
            Gene::newRandom(),
            Gene::newRandom(),
            Gene::newRandom(),
            Gene::newRandom(),
        ];

        Bob {
            position_x: x as f32,
            position_y: y as f32,
            genes,
            Age: 0.,
        }
    }

    fn MvU(&mut self, grid: &mut Vec<Vec<Option<Bob>>>, distance: i32) {
        self.moveTo(
            self.position_x as usize,
            (self.position_y - 1.) as usize,
            grid,
        );
    }

    fn MvD(&mut self, grid: &mut Vec<Vec<Option<Bob>>>, distance: i32) {
        self.moveTo(
            self.position_x as usize,
            (self.position_y + 1.) as usize,
            grid,
        );
    }

    fn MvR(&mut self, grid: &mut Vec<Vec<Option<Bob>>>, distance: i32) {
        self.moveTo(
            (self.position_x + 1.) as usize,
            self.position_y as usize,
            grid,
        );
    }

    fn MvL(&mut self, grid: &mut Vec<Vec<Option<Bob>>>, distance: i32) {
        self.moveTo(
            (self.position_x - 1.) as usize,
            self.position_y as usize,
            grid,
        );
    }

    fn execute_action(
        &mut self,
        action: outputNeurons,
        grid: &mut Vec<Vec<Option<Bob>>>,
        distance: i32,
    ) {
        match action {
            outputNeurons::MvUp => {
                self.MvU(grid, distance);
            }
            outputNeurons::MvD => {
                self.MvD(grid, distance);
            }
            outputNeurons::MvR => {
                self.MvR(grid, distance);
            }
            outputNeurons::MvL => {
                self.MvL(grid, distance);
            }
            outputNeurons::MvRandom => {
                self.MvRandom(grid, distance);
            }
            outputNeurons::Halt => {}
            _ => {
                println!("invalid move enum");
            }
        }
    }

    fn moveTo(&mut self, new_x: usize, new_y: usize, grid: &mut Vec<Vec<Option<Bob>>>) -> bool {
        if new_x < GRID_SIZE && new_y < GRID_SIZE && grid[new_y][new_x].is_none() {
            let x = self.position_x as usize;
            let y = self.position_y as usize;

            grid[y][x] = None;
            grid[new_y][new_x] = Some(self.clone());
            self.position_x = new_x as f32;
            self.position_y = new_y as f32;
            true 
        } else {
            false 
        }
    }

    fn MvRandom(&mut self, grid: &mut Vec<Vec<Option<Bob>>>, distance: i32) {
        let mut rng = rand::thread_rng();
        let rand_num = rng.gen_range(0..4);

        match rand_num {
            0 => self.MvU(grid, distance),
            1 => self.MvD(grid, distance),
            2 => self.MvR(grid, distance),
            3 => self.MvL(grid, distance),
            _ => {
                println!("invalid option");
            }
        }
    }
    fn countNeighbors(&self, grid: &Vec<Vec<Option<Bob>>>) -> u8 {
        let mut count: u8 = 0;

        let neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ];

        let x = self.position_x as usize;
        let y = self.position_y as usize;

        for (dx, dy) in &neighbors {
            let new_x = x as isize + dx;
            let new_y = y as isize + dy;

            if new_x >= 0
                && new_x < grid.len() as isize
                && new_y >= 0
                && new_y < grid[0].len() as isize
            {
                if grid[new_x as usize][new_y as usize].is_some() {
                    count += 1;
                }
            }
        }
        count
    }

    fn getSensoryValue(
        &self,
        sense: inputNeurons,
        grid_size: usize,
        grid: &Vec<Vec<Option<Bob>>>,
    ) -> f32 {
        match sense {
            inputNeurons::Age => {
                let percentage_completion = self.Age / 100.0;
                percentage_completion.min(1.0)
            }
            inputNeurons::BorderDistanceBot => {
                let distance_to_bot = self.position_y / (grid_size as f32);
                distance_to_bot.max(0.0).min(1.0)
            }
            inputNeurons::BorderDistanceTop => {
                let distance_to_top = (grid_size as f32 - self.position_y) / (grid_size as f32);
                distance_to_top.max(0.0).min(1.0)
            }
            inputNeurons::BorderDistanceLeft => {
                let distance_to_left = self.position_x / (grid_size as f32);
                distance_to_left.max(0.0).min(1.0)
            }
            inputNeurons::BorderDistanceRight => {
                let distance_to_right = (grid_size as f32 - self.position_x) / (grid_size as f32);
                distance_to_right.max(0.0).min(1.0)
            }
            inputNeurons::countNeighbors => self.countNeighbors(grid) as f32,
        }
    }

    fn calcAndExecMovement(
        &mut self,
        sense: inputNeurons,
        grid: &mut Vec<Vec<Option<Bob>>>,
        grid_size: usize,
        weight: f64,
        action: outputNeurons,
    ) {
        const MOVEMENT_AMOUNT: f64 = 1.0;
        const SCALING_FACTOR: f64 = 100.0;

        let raw_movement = (weight * self.getSensoryValue(sense, grid_size, grid) as f64)
            .round()
            .max(-MOVEMENT_AMOUNT)
            .min(MOVEMENT_AMOUNT);

        let percentage_movement = (raw_movement / MOVEMENT_AMOUNT * SCALING_FACTOR).round();

        println!("raw movement {}", raw_movement);
        println!("percentage movement {}", percentage_movement);

        self.execute_action(action, grid, percentage_movement as i32);
    }

    fn select_best_action(
        &self,
        grid: &mut Vec<Vec<Option<Bob>>>,
        grid_size: usize,
    ) -> outputNeurons {
        let mut best_action = outputNeurons::Halt; 
        let mut best_score = f64::NEG_INFINITY; 

        for gene in &self.genes {
            if let Some(input_neuron) = gene.source_neuron() {
                if let Some(output_neuron) = gene.destination_neuron() {
                    let weight = gene.weight();

                    let sensory_value = self.getSensoryValue(input_neuron, grid_size, grid);

                    let score = (weight * sensory_value) as f64;

                    if score > best_score {
                        best_score = score;
                        best_action = output_neuron;
                    }
                }
            }
        }

        best_action
    }

    fn execute_best_action(&mut self, grid: &mut Vec<Vec<Option<Bob>>>, grid_size: usize) {
        let best_action = self.select_best_action(grid, grid_size);
        self.execute_action(best_action, grid, 1);
    }

    fn reproduce(&self, partner: &Bob) -> Bob {
        let mut rng = rand::thread_rng();
        let mut new_genes = Vec::new();

        for (gene1, gene2) in self.genes.iter().zip(partner.genes.iter()) {
            let crossover_point = rng.gen_range(0..GENE_SIZE);

            let new_value = if rng.gen::<f64>() < 0.5 {
                (gene1.value & !(u32::MAX << crossover_point))
                    | (gene2.value & (u32::MAX << crossover_point))
            } else {
                (gene2.value & !(u32::MAX << crossover_point))
                    | (gene1.value & (u32::MAX << crossover_point))
            };

            let mutation_rate = 0.0;
            if rng.gen::<f64>() < mutation_rate {
                let mutation_bit = 1 << rng.gen_range(0..GENE_SIZE);
                new_genes.push(Gene {
                    value: new_value ^ mutation_bit,
                });
            } else {
                new_genes.push(Gene { value: new_value });
            }
        }

        let x = rng.gen_range(0..GRID_SIZE);
        let y = rng.gen_range(0..GRID_SIZE);

        Bob {
            position_x: x as f32,
            position_y: y as f32,
            genes: new_genes,
            Age: 0.0,
        }
    }

    fn printInfo(&self) {
        println!("Bob info:");
        println!("pos x {}", self.position_x);
        println!("pos y {}", self.position_y);

        for gene in &self.genes {
            println!("is positive: {}", gene.is_positive());
            println!(
                "Source: {}",
                gene.source_neuron().unwrap_or(inputNeurons::Age)
            );
            println!(
                "destination: {}",
                gene.destination_neuron().unwrap_or(outputNeurons::Halt)
            );
            println!("weight {}", gene.weight());
        }
    }
}

struct GameState {
    grid: Vec<Vec<Option<Bob>>>,
    all_bobs: Vec<Bob>,
}

impl GameState {
    fn new(population: i32) -> Self {
        let mut grid = vec![vec![None; GRID_SIZE]; GRID_SIZE];
        let mut all_bobs = Vec::new();

        for _ in 0..population {
            let mut x;
            let mut y;

            loop {
                let mut rng = rand::thread_rng();

                x = rng.gen_range(0..GRID_SIZE);
                y = rng.gen_range(0..GRID_SIZE);

                if grid[y][x].is_none() {
                    break;
                }
            }

            let bob = Bob::new(x, y);

            println!("og x {}", x);
            println!("og y {}", y);

            grid[y][x] = Some(bob.clone());
            all_bobs.push(bob);
        }

        GameState { grid, all_bobs }
    }

    fn moveAllRandom(&mut self) {
        for bob in &mut self.all_bobs {
            bob.execute_action(outputNeurons::MvRandom, &mut self.grid, 1);
            bob.Age += 1.;
        }
    }

    fn updatePositions(&mut self) {
        for bob in &mut self.all_bobs {
            bob.execute_best_action(&mut self.grid, GRID_SIZE);
            //bob.printInfo();
            bob.Age += 1.;
        }
    }

    fn display(&self) {
        for row in &self.grid {
            for cell in row {
                match cell {
                    Some(_) => {
                        print!("[#]");
                    }
                    None => {
                        print!("[ ]");
                    }
                }
            }
            println!();
        }
    }

    fn evaluate_and_select_survivors(&mut self) {
        println!("Number of bobs before evaluation: {}", self.all_bobs.len());

        let split_columnL = GRID_SIZE / 4;
        let split_columnR = (GRID_SIZE / 4) * 3; 

        let mut left_side_bobs = Vec::new();
        let mut right_side_bobs = Vec::new();

        for bob in self.all_bobs.drain(..) {
            let x = bob.position_x as usize;
            if x < split_columnL || x > split_columnR {
                left_side_bobs.push(bob);
            } else {
                right_side_bobs.push(bob);
            }
        }

        for row in &mut self.grid {
            for cell in row.iter_mut() {
                *cell = None;
            }
        }

        for bob in &left_side_bobs {
            let x = bob.position_x as usize;
            let y = bob.position_y as usize;
            self.grid[y][x] = Some(bob.clone());
        }

        self.all_bobs = left_side_bobs;

        println!("Number of bobs after evaluation: {}", self.all_bobs.len());
    }

    fn mutate_genes(&mut self) {
        let mutation_rate = 0.1;

        for bob in &mut self.all_bobs {
            bob.printInfo();
        }
    }

    fn repopulate(&mut self, population: i32) {
        let mut new_bobs = Vec::new();

        while new_bobs.len() < population as usize {
            let parent1_index = rand::thread_rng().gen_range(0..self.all_bobs.len());
            let parent2_index = rand::thread_rng().gen_range(0..self.all_bobs.len());
            let parent1 = &self.all_bobs[parent1_index];
            let parent2 = &self.all_bobs[parent2_index];

            let offspring = parent1.reproduce(parent2);

            new_bobs.push(offspring);
        }

        self.all_bobs = new_bobs;
    }

    fn run_generations(&mut self, num_generations: usize, steps_per_generation: usize) {
        for _ in 0..num_generations {
            println!("new generation");
            std::thread::sleep(std::time::Duration::from_millis(1000));
            for i in 0..steps_per_generation {
                println!("\n\n\n\n\n\n\n\n\n\n\n\n new step");
                self.updatePositions();
                self.display();
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            //self.displayGenes();
            self.evaluate_and_select_survivors();
            //self.mutate_genes();
            self.repopulate(100);
        }
    }

    fn displayGenes(&mut self){
        for bob in &mut self.all_bobs {
            println!("~~~~~~~~~~~~\nBob:");
            bob.printInfo();
        }
    }
}

fn main() {
    println!("\x1B[2J\x1B[H");
    let mut state = GameState::new(300);

    println!("Initial State:");
    state.display();

    state.run_generations(120, 300);
}
