//date: 2022-04-13T17:11:10Z
//url: https://api.github.com/gists/ceec7620684ecb9a49e3a0c422c8a560
//owner: https://api.github.com/users/jeyakeerthanan

package com.company;

public class Mobile {
    private  String camera;
    private String  airpod;
    private String mobileCover;
    private String temperedGlass;
    private String processor;

    public Mobile(Builder builder) {
        this.processor=builder.processor;
        this.airpod=builder.airpod;
        this.camera=builder.camera;
        this.mobileCover=builder.mobileCover;
        this.temperedGlass=builder.temperedGlass;
    }


    static class Builder{
        private  String camera;
        private String  airpod;
        private String mobileCover;
        private String temperedGlass;
        private String processor;

        public Mobile build(){
            return new Mobile(this);
        }

        public Builder (String processor){
            this.processor=processor;
        }

        public Builder airpod (String airpod){
            this.airpod=airpod;
            return this;
        }

        public Builder mobileCover(String mobileCover){
            this.mobileCover=mobileCover;
            return this;
        }

        public Builder temperedGlass (String temperedGlass){
            this.temperedGlass=temperedGlass;
            return this;
        }
        public Builder camera (String camera){
            this.camera=camera;
            return this;
        }

        @Override
        public String toString() {
            return "Builder{" +
                    "camera='" + camera + '\'' +
                    ", airpod='" + airpod + '\'' +
                    ", mobileCover='" + mobileCover + '\'' +
                    ", temperedGlass='" + temperedGlass + '\'' +
                    ", processor='" + processor + '\'' +
                    '}';
        }
    }

    @Override
    public String toString() {
        return "Mobile{" +
                "camera='" + camera + '\'' +
                ", airpod='" + airpod + '\'' +
                ", mobileCover='" + mobileCover + '\'' +
                ", temperedGlass='" + temperedGlass + '\'' +
                ", processor='" + processor + '\'' +
                '}';
    }
}
