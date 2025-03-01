function SmoothVar(value, min, max){
    this.value = value
    this.min = min
    this.max = max
    this.target = value

    let self = this
    
    this.update = function(value){
        self.target = Math.max(self.min, value)
        self.target = Math.min(self.max, self.target)
        self.value += (self.target - self.value) * 0.2
    }

    this.update_add = function(delta){
        self.update(self.target + delta)
    }
}

export {SmoothVar}